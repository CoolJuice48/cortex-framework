# Resolve forking errors in LLM calls
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
# Force reload modules to avoid caching issues
if 'graph' in sys.modules:
    del sys.modules['graph']
if 'classifier' in sys.modules:
    del sys.modules['classifier']

from embed import Embedder
from graph import KnowledgeGraph, check_and_split_domains, query_graph
from extraction import QuestionExtractor, insert_question_smart
from classifier import DomainClassifier
from loader import DocumentLoader
from config import ANTHROPIC_API_KEY

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "./data"
LOG_DIR = "./log"
OUTPUT_DIR = "./output"
EMBEDDINGS_DIR = "./embeddings"
QUESTIONS_DIR = "./questions"

# Start small for testing
FILE_LIMIT = 1
ROW_LIMIT = 100   # More documents to trigger splits
DOMAIN = "automotive"  # Start broad

# Log file name
TEST_NUM = "test10"
TAG = "stored_questions"
from datetime import datetime
TIMESTAMP = datetime.now().strftime('%m-%d-%y_%I:%M%p')
LOG_FILE = f"{TEST_NUM}_{TIMESTAMP}_{TAG}.log"

# =============================================================================
# INITIALIZE
# =============================================================================

print("Initializing components...")
embedder = Embedder()
graph = KnowledgeGraph(embedder)
extractor = QuestionExtractor(ANTHROPIC_API_KEY)
classifier = DomainClassifier(ANTHROPIC_API_KEY)  # NEW!
loader = DocumentLoader(
   embedder,
   log_file=LOG_FILE,
   log_dir=LOG_DIR,
   output_dir=OUTPUT_DIR,
   embeddings_dir=EMBEDDINGS_DIR,
   questions_dir=QUESTIONS_DIR
)

# =============================================================================
# LOAD & INGEST WITH DOMAIN CLASSIFICATION
# =============================================================================

print(f"\nLoading up to {ROW_LIMIT} TSBs from {DATA_DIR}...")

# Try to load cached embeddings first
EMBEDDING_CACHE = f"embeddings_{ROW_LIMIT}docs.pkl"
cached_embeddings = loader.load_embeddings(EMBEDDING_CACHE)

if cached_embeddings:
    print(f"Found cached embeddings for {len(cached_embeddings)} documents!")
    print("Loading from cache...")
    
    # Load documents WITHOUT embedding (faster)
    documents = loader.load_from_directory(
        directory=DATA_DIR,
        file_extensions=['.tsv'],
        limit=FILE_LIMIT,
        row_limit=ROW_LIMIT
    )
    
    # Apply cached embeddings
    for doc in documents:
        if doc.id in cached_embeddings:
            doc.embedding = cached_embeddings[doc.id]['embedding']
            print(f"  Restored embedding for {doc.id}")
    
    print(f"Loaded {len(documents)} documents from cache")
else:
    print("No cache found, embedding documents...")
    
    # Load and embed normally
    documents = loader.load_from_directory(
        directory=DATA_DIR,
        file_extensions=['.tsv'],
        limit=FILE_LIMIT,
        row_limit=ROW_LIMIT
    )
    
    # Save embeddings for next time
    print(f"Saving embeddings to cache...")
    loader.save_embeddings(documents, EMBEDDING_CACHE)
    print(f"Cached {len(documents)} document embeddings")

print(f"Ready to process {len(documents)} documents")

# Track stats
total_stats = {
   'docs_processed': 0,
   'questions_extracted': 0,
   'questions_added': 0,
   'duplicates': 0,
   'errors': 0,
   'domain_splits': 0
}

all_questions = []

# Process documents with domain splitting checks
for i, doc in enumerate(documents, start=1):
    try:
        # Add document to graph
        graph.add_document(doc)
        
        # STEP 0: Identify domain(s) for this document
        existing_domains = list(graph.domains.keys())
        doc_domains = classifier.identify_domain(doc.text, existing_domains)
        
        loader.logger.info(f"\nDoc {i}/{len(documents)}: {doc.id}")
        loader.logger.info(f"  Identified domains: {', '.join(doc_domains)}")
        
        # STEP 1: Check if doc answers existing questions (filter by domain)
        existing_answered = []
        for domain in doc_domains:
            relevant_questions = [
                q for q in graph.questions.values()
                if domain in q.domains
            ]
            
            for question in relevant_questions:
                from structs import cosine_similarity
                similarity = cosine_similarity(question.embedding, doc.embedding)
                if similarity > 0.75:
                    question.answer_documents.append(doc)
                    existing_answered.append(question.text)
        
        if existing_answered:
            loader.logger.info(f"  Document answers {len(existing_answered)} existing questions")
        
        # STEP 2: Extract questions for each domain (with caching)
        for domain in doc_domains:
            # Try to load from cache first
            questions = loader.load_questions_cache(doc.id, graph.documents)

            if questions:
                loader.logger.info(f"  Loaded {len(questions)} questions from cache")
            else:
                # Extract fresh questions
                loader.logger.info(f"  Extracting questions with LLM...")
                questions = extractor.extract_from_text(
                    doc.text,
                    doc,
                    num_questions=5,
                    existing_questions=all_questions + existing_answered
                )
                # Cache new questions
                loader.save_questions_cache(doc.id, questions)
                loader.logger.info(f"  Cached {len(questions)} questions")
            
            # Add questions to graph (assign all domains to each question)
            for q_text, answer_docs in questions:
                # First domain for insertion, question will have all domains
                result = insert_question_smart(graph, q_text, answer_docs, doc_domains[0])
                if result:
                    for domain in doc_domains[1:]:
                        if domain not in result.domains:
                            result.domains.append(domain)

                    total_stats['questions_added'] += 1
                    all_questions.append(q_text)
                    loader.logger.info(f"  Added [{', '.join(result.domains)}]: {q_text}")
                else:
                    total_stats['duplicates'] += 1
                    loader.logger.info(f"  Duplicate: {q_text}")
            
            total_stats['questions_extracted'] += len(questions)
        
        total_stats['docs_processed'] += 1
        
        # STEP 3: Check for domain splits every 20 documents
        if i % 20 == 0:
            loader.logger.info(f"\n{'='*60}")
            loader.logger.info(f"CHECKING FOR DOMAIN SPLITS (after {i} documents)")
            loader.logger.info(f"{'='*60}")
            
            pre_split_domains = len(graph.domains)
            check_and_split_domains(graph, classifier, threshold_questions=10)
            post_split_domains = len(graph.domains)
            
            splits_this_round = post_split_domains - pre_split_domains
            if splits_this_round > 0:
                total_stats['domain_splits'] += splits_this_round
                loader.logger.info(f"Domains after split: {list(graph.domains.keys())}")
        
    except Exception as e:
        loader.logger.error(f"ERROR processing {doc.id}: {e}", exc_info=True)
        total_stats['errors'] += 1
        continue

# Final domain split check
print(f"\n{'='*60}")
print(f"FINAL DOMAIN SPLIT CHECK")
print(f"{'='*60}")
check_and_split_domains(graph, classifier, threshold_questions=10)

# =============================================================================
# DIAGNOSTIC: CHECK DOMAIN COHERENCE
# =============================================================================

from graph import analyze_domain_coherence

print(f"\n{'='*60}")
print(f"DOMAIN COHERENCE ANALYSIS")
print(f"{'='*60}")

for domain in graph.domains.keys():
    analysis = analyze_domain_coherence(graph, domain)
    print(f"\nDomain: {domain}")
    print(f"  Should split: {analysis.get('should_split', 'N/A')}")
    print(f"  Variance: {analysis.get('variance', 'N/A')}")
    print(f"  Avg similarity: {analysis.get('avg_similarity', 'N/A')}")
    print(f"  Num questions: {analysis.get('num_questions', 0)}")
    
    if 'reason' in analysis:
        print(f"  Reason: {analysis['reason']}")

# =============================================================================
# DIAGNOSTIC: CHECK DOMAIN CLASSIFICATION
# =============================================================================

print(f"\n{'='*60}")
print(f"DOMAIN CLASSIFICATION DIAGNOSTIC")
print(f"{'='*60}")

# Sample 10 random documents and show their classifications
import random
sample_docs = random.sample(documents, min(10, len(documents)))

for doc in sample_docs:
    existing_domains = list(graph.domains.keys())
    doc_domains = classifier.identify_domain(doc.text, existing_domains)
    
    print(f"\nDoc: {doc.id}")
    print(f"Text preview: {doc.text[:200]}...")
    print(f"Classified as: {doc_domains}")

# =============================================================================
# RESULTS
# =============================================================================

print(f"\n{'='*60}")
print(f"INGESTION COMPLETE")
print(f"{'='*60}")
print(f"Documents processed: {total_stats['docs_processed']}")
print(f"Questions extracted: {total_stats['questions_extracted']}")
print(f"Questions added: {total_stats['questions_added']}")
print(f"Duplicates caught: {total_stats['duplicates']}")
print(f"Errors: {total_stats['errors']}")
print(f"Domain splits: {total_stats['domain_splits']}")

if total_stats['questions_extracted'] > 0:
   dedup_rate = total_stats['duplicates'] / total_stats['questions_extracted'] * 100
   print(f"Deduplication rate: {dedup_rate:.1f}%")

print(f"\nFinal graph size: {len(graph.questions)} questions")
print(f"Total domains: {len(graph.domains)}")
print(f"Domains: {list(graph.domains.keys())}")

# =============================================================================
# ANALYZE DOMAIN STRUCTURE
# =============================================================================

print(f"\n{'='*60}")
print(f"DOMAIN ANALYSIS")
print(f"{'='*60}")

for domain_name in graph.domains.keys():
    domain_questions = [q for q in graph.questions.values() if domain_name in q.domains]
    root_questions = graph.get_question_roots(domain_name)
    
    print(f"\n{domain_name.upper()}:")
    print(f"  Total questions: {len(domain_questions)}")
    print(f"  Root questions: {len(root_questions)}")
    
    # Show sample root questions
    if root_questions:
        print(f"  Sample roots:")
        for root in root_questions[:3]:
            print(f"    • {root.text}")
            if root.children:
                print(f"      └─ {len(root.children)} children")

# =============================================================================
# TEST QUERIES ACROSS DOMAINS
# =============================================================================

print(f"\n{'='*60}")
print(f"TESTING QUERIES")
print(f"{'='*60}")

test_queries = [
   "What causes transmission problems?",
   "How do you fix brake issues?",
   "What are common engine problems?",
]

for query in test_queries:
   print(f"\nQuery: '{query}'")
   results = query_graph(graph, query, top_k=3)
   
   if results:
      for i, (q, score) in enumerate(results, 1):
         domains_str = ', '.join(q.domains)
         print(f"  {i}. [{domains_str}] (score: {score:.3f}) {q.text}")
   else:
      print("  No results found")

# =============================================================================
# SAVE STATE
# =============================================================================

print(f"\n{'='*60}")
print(f"SAVING STATE")
print(f"{'='*60}")

import pickle

graph_path = os.path.join(OUTPUT_DIR, f"graph_domains_{ROW_LIMIT}docs.pkl")
with open(graph_path, 'wb') as f:
   pickle.dump(graph, f)
print(f"Saved graph to: {graph_path}")

# Save embeddings for reuse (NEW!)
print(f"Saving embeddings to cache...")
loader.save_embeddings(documents, EMBEDDING_CACHE)
print(f"Cached {len(documents)} document embeddings")

# Save domain structure
domain_summary = {
    'total_domains': len(graph.domains),
    'domains': {
        name: {
            'total_questions': len([q for q in graph.questions.values() if name in q.domains]),
            'root_questions': len(roots)
        }
        for name, roots in graph.domains.items()
    }
}

domain_path = os.path.join(OUTPUT_DIR, f"domains_{ROW_LIMIT}docs.json")
import json
with open(domain_path, 'w') as f:
    json.dump(domain_summary, f, indent=2)
print(f"Saved domain summary to: {domain_path}")

print(f"\nDone! Check {LOG_DIR}/{LOG_FILE} for details.")