# test_refactor.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from embed import Embedder
from graph import KnowledgeGraph
from extraction import QuestionExtractor
from classifier import DomainClassifier
from loader import DocumentLoader
from structs import Document, Answer, Question, Domain
from config import ANTHROPIC_API_KEY

print("="*60)
print("REFACTOR TEST - Simple Verification")
print("="*60)

# =============================================================================
# STEP 1: Initialize Components
# =============================================================================
print("\n[1/5] Initializing components...")

embedder = Embedder()
graph = KnowledgeGraph(embedder)
extractor = QuestionExtractor(ANTHROPIC_API_KEY, embedder)
classifier = DomainClassifier(ANTHROPIC_API_KEY)

print("✓ All components initialized")

# =============================================================================
# STEP 2: Create Test Documents
# =============================================================================
print("\n[2/5] Creating test documents...")

test_docs = [
   Document(
      id="test_doc_1",
      text="The hydraulic brake system uses fluid pressure to stop the vehicle. Regular brake fluid changes prevent moisture buildup.",
      metadata={'source': 'test', 'domain': 'brakes'}
   ),
   Document(
      id="test_doc_2", 
      text="Automatic transmission fluid should be changed every 30,000 miles. Low fluid levels can cause shifting problems.",
      metadata={'source': 'test', 'domain': 'transmission'}
   ),
   Document(
      id="test_doc_3",
      text="Engine coolant prevents overheating and protects against freezing. Check coolant level monthly.",
      metadata={'source': 'test', 'domain': 'engine'}
   )
]

# Add documents to graph
for doc in test_docs:
   graph.add_document(doc)

print(f"✓ Created and added {len(test_docs)} test documents")

# =============================================================================
# STEP 3: Test Domain Classification
# =============================================================================
print("\n[3/5] Testing domain classification...")

doc_domains = classifier.identify_domain(test_docs[0].text, [])
print(f"✓ Doc 1 classified as: {doc_domains}")

# =============================================================================
# STEP 4: Test Question Extraction
# =============================================================================
print("\n[4/5] Testing question extraction...")

qa_pairs = extractor.extract_from_text(
   test_docs[0].text,
   test_docs[0],
   num_questions=2,
   existing_questions=[]
)

print(f"✓ Extracted {len(qa_pairs)} question-answer pairs")
for q_text, answer in qa_pairs:
   print(f"  Q: {q_text}")
   print(f"  A: {answer.text[:50]}..." if len(answer.text) > 50 else f"  A: {answer.text}")
   print(f"  Answer has {len(answer.source_documents)} source doc(s)")
   print(f"  Answer embedded: {answer.embedding is not None}")

# =============================================================================
# STEP 5: Test Adding Questions to Graph
# =============================================================================
print("\n[5/5] Testing graph operations...")

from graph import insert_question_smart

# Add first question
q_text, answer = qa_pairs[0]
result = insert_question_smart(
   graph,
   q_text,
   answer,
   domain_names=["brake_system"]
)

if result:
   print(f"✓ Added question to graph: '{q_text}'")
   print(f"  Question ID: {result.id}")
   print(f"  Domains: {result.domains}")
   print(f"  Answers: {len(result.answers)}")
else:
   print("✗ Failed to add question")

# Check graph state
print(f"\n✓ Graph state:")
print(f"  Total documents: {len(graph.documents)}")
print(f"  Total questions: {len(graph.questions)}")
print(f"  Total domains: {len(graph.domains)}")
print(f"  Domain names: {list(graph.domains.keys())}")

# Check domain structure
for domain_name, domain in graph.domains.items():
   print(f"\n✓ Domain '{domain_name}':")
   print(f"  Questions: {len(domain.questions)}")
   print(f"  Parent: {domain.parent_domain.name if domain.parent_domain else 'None'}")
   print(f"  Subdomains: {len(domain.subdomains)}")

# =============================================================================
# FINAL VERIFICATION
# =============================================================================
print("\n" + "="*60)
print("VERIFICATION CHECKLIST")
print("="*60)

checks = {
   "Documents added": len(graph.documents) == 3,
   "Questions created": len(graph.questions) > 0,
   "Domains created": len(graph.domains) > 0,
   "Answers have embeddings": qa_pairs[0][1].embedding is not None,
   "Answers have source docs": len(qa_pairs[0][1].source_documents) > 0,
   "Question has domains": len(result.domains) > 0 if result else False,
   "Question has answers": len(result.answers) > 0 if result else False,
}

all_passed = True
for check, passed in checks.items():
   status = "✓" if passed else "✗"
   print(f"{status} {check}")
   if not passed:
      all_passed = False

print("\n" + "="*60)
if all_passed:
   print("🎉 ALL CHECKS PASSED! Refactor successful!")
else:
   print("⚠️  Some checks failed - review output above")
print("="*60)