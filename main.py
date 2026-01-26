from embed import Embedder
from graph import KnowledgeGraph
from extraction import QuestionExtractor
from loader import DocumentLoader
from config import API_KEY
from datetime import datetime
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "./data"
LOG_DIR = "./log"
OUTPUT_DIR = "./output"

# Start small for testing
FILE_LIMIT = 20  # Number of TSBs to process
ROW_LIMIT = 50   # Number of rows to process per document
DOMAIN = "automotive"  # Start broad, not just transmission

# Log file name
TEST_NUM = "test2"
TAG = "TSB"
TIMESTAMP = datetime.now().strftime('%m-%d-%y_%I:%M%p')  # Auto-generates
LOG_FILE = f"{TEST_NUM}_{TIMESTAMP}_{TAG}.log"
# Result: "test2_01-26-26_01:05AM_TSB.log"

# =============================================================================
# INITIALIZE
# =============================================================================

print("Initializing components...")
embedder = Embedder()
graph = KnowledgeGraph(embedder)
extractor = QuestionExtractor(API_KEY)
loader = DocumentLoader(
   embedder,
   log_file=LOG_FILE,
   log_dir=LOG_DIR,
   output_dir=OUTPUT_DIR
)

# =============================================================================
# LOAD & INGEST
# =============================================================================

print(f"\nLoading up to {FILE_LIMIT} files from {DATA_DIR}...")

# Option A: Load ALL TSBs (test self-organization)
stats = loader.load_and_ingest(
   graph=graph,
   extractor=extractor,
   directory=DATA_DIR,
   domain=DOMAIN,
   file_extensions=['.tsv'],
   limit=FILE_LIMIT,
   row_limit=ROW_LIMIT
)

# Option B: Filter to transmission only (uncomment to use)
# def is_transmission(filepath: str) -> bool:
#     return 'transmission' in filepath.lower() or 'TRANSMISSION' in filepath.lower()
# 
# stats = loader.load_and_ingest(
#     graph=graph,
#     extractor=extractor,
#     directory=DATA_DIR,
#     domain="transmission",
#     file_extensions=['.tsv'],
#     filter_fn=is_transmission,
#     limit=LIMIT
# )

# =============================================================================
# RESULTS
# =============================================================================

print(f"\n{'='*60}")
print(f"INGESTION COMPLETE")
print(f"{'='*60}")
print(f"Documents processed: {stats['docs_processed']}")
print(f"Questions extracted: {stats['questions_extracted']}")
print(f"Questions added: {stats['questions_added']}")
print(f"Duplicates caught: {stats['duplicates']}")
print(f"Errors: {stats['errors']}")

if stats['questions_extracted'] > 0:
   dedup_rate = stats['duplicates'] / stats['questions_extracted'] * 100
   print(f"Deduplication rate: {dedup_rate:.1f}%")

print(f"\nFinal graph size: {len(graph.questions)} questions")
print(f"Root questions: {len(graph.get_question_roots(DOMAIN))}")

# =============================================================================
# ANALYZE GRAPH STRUCTURE
# =============================================================================

print(f"\n{'='*60}")
print(f"GRAPH HIERARCHY (Top 10 roots)")
print(f"{'='*60}")

roots = graph.get_question_roots(DOMAIN)
for i, root in enumerate(roots[:10], 1):
   print(f"\n{i}. {root.text}")
   if root.children:
      print(f"   └─ {len(root.children)} children:")
      for child in root.children[:3]:
         print(f"      • {child.text}")
      if len(root.children) > 3:
         print(f"      ... and {len(root.children) - 3} more")

# =============================================================================
# TEST QUERIES
# =============================================================================

print(f"\n{'='*60}")
print(f"TESTING QUERIES")
print(f"{'='*60}")

from graph import query_graph

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
         print(f"  {i}. (score: {score:.3f}) {q.text}")
   else:
      print("  No results found")

# =============================================================================
# SAVE STATE (optional)
# =============================================================================

print(f"\n{'='*60}")
print(f"SAVING STATE")
print(f"{'='*60}")

# Save the graph for later use
import pickle

graph_path = os.path.join(OUTPUT_DIR, f"graph_{DOMAIN}_{FILE_LIMIT}docs.pkl")
with open(graph_path, 'wb') as f:
   pickle.dump(graph, f)
print(f"Saved graph to: {graph_path}")

print(f"\nDone! Check {LOG_DIR}/{LOG_FILE}.log for details.")