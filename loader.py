# ------------------------------------------------------------------ #
# Cortex imports
from embed import Embedder
from graph import KnowledgeGraph, cosine_similarity
from extraction import QuestionExtractor, insert_question_smart
from structs import Document
from classifier import DomainClassifier
# ------------------------------------------------------------------ #
# Python imports
from pathlib import Path
from typing import List, Optional, Callable, Tuple
from datetime import datetime
import pickle
import os
import logging
import json
import csv
# ------------------------------------------------------------------ #

"""
Standardized document loading & ingestion
"""
class DocumentLoader:
   def __init__(
         self,
         embedder: Embedder,
         log_file: str=None,
         log_dir: str='log',
         output_dir: str='output',
         embeddings_dir: str='embeddings',
         questions_dir: str='questions'
      ):
      self.embedder = embedder
      self.output_dir = output_dir
      self.embeddings_dir = embeddings_dir
      self.questions_dir = questions_dir

      os.makedirs(output_dir, exist_ok=True)
      os.makedirs(log_dir, exist_ok=True)
      os.makedirs(embeddings_dir, exist_ok=True)
      os.makedirs(questions_dir, exist_ok=True)

      if log_file is None:
         log_file = f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

      log_path = os.path.join(log_dir, log_file)
      
      # Create logger
      self.logger = logging.getLogger(__name__)
      self.logger.setLevel(logging.INFO)

      # Clear existing handlers
      self.logger.handlers = []

      # File handler
      file_handler = logging.FileHandler(log_path)
      file_handler.setLevel(logging.INFO)

      # Console handler (optional - still print to console)
      console_handler = logging.StreamHandler()
      console_handler.setLevel(logging.INFO)

      # Format
      formatter = logging.Formatter(
         '%(asctime)s - %(levelname)s - %(message)s',
         datefmt='%Y-%m-%d %H:%M'
      )
      file_handler.setFormatter(formatter)
      console_handler.setFormatter(formatter)

      # Add handlers
      self.logger.addHandler(file_handler)
      self.logger.addHandler(console_handler)

      # Record initialization
      self.logger.info(f"Logging to: {log_path}")

   """
   Load documents from a directory.
   
   Args:
      directory: Path to directory containing documents
      file_extensions: Which file types to load
      filter_fn: Optional function to filter files (receives filename)
      limit: Maximum number of documents to load
   
   Returns:
      List of Document objects
   """
   def load_from_directory(
      self,
      directory: str,
      file_extensions: List[str] = ['.txt', '.json'],
      filter_fn: Optional[Callable[[str], bool]] = None,
      limit: Optional[int] = None,
      row_limit: int=None
   ) -> List[Document]:
      documents = []
      directory_path = Path(directory)

      if not directory_path.exists():
         raise ValueError(f"Directory not found: {directory}")
      
      # Get matching files
      files = []
      for ext in file_extensions:
         files.extend(directory_path.glob(f"*{ext}"))
         
      # Apply filter if needed
      if filter_fn:
         files = [f for f in files if filter_fn(str(f))]
      
      # Apply limit
      if limit:
         files = files[:limit]

      print(f"Found {len(files)} documents to load")

      # Load each file
      for filepath in files:
         try:
            docs = self._load_single_file(filepath, row_limit=row_limit)
            if docs:
               documents.extend(docs)
         except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}", exc_info=True)
            continue

      self.logger.info(f"Successfully loaded {len(documents)} documents")
      return documents
   
   """
   Load a single file and creates Document(s) for it
   Returns a list, one document or multiple (for CSV/TSV)
   Text files return a single item list
   """
   def _load_single_file(self, filepath: Path, row_limit: int=None) -> Optional[List[Document]]:
      extension = filepath.suffix.lower()
      # Handle tabular data
      if extension in ['.csv', '.tsv']:
         return self._load_tabular_file(filepath, extension, row_limit=row_limit)
      
      # Handle text files
      elif extension in ['.txt', '.json', '.md']:
         return self._load_text_file(filepath)
      
      else:
         print(f"Unsupported file type: {extension}")
         return None
   
   """
   Loads CSV/TSB file, each row becoming a document
   Assumes first row contains headers
   """
   def _load_tabular_file(self, filepath: Path, extension: str, row_limit: int=None) -> List[Document]:
      delimiter = '\t' if extension == '.tsv' else ','
      documents = []

      with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
         reader = csv.DictReader(f, delimiter=delimiter)

         for i, row in enumerate(reader):
            # Apply row limit if specified
            if row_limit and i >= row_limit:
               self.logger.info(f"Reached row limit ({row_limit}), stopping")
               break

            # Convert row to text (concatenate all fields)
            row_text = ' | '.join(f"{k}: {v}" for k, v in row.items() if v)

            if not row_text.strip():
               continue

            metadata = self._extract_metadata(filepath)
            metadata['line_number'] = i + 1
            metadata['is_multiline'] = True

            doc = Document(
               id=f"{filepath.stem}_row_{i+1}", # WHAT SHOULD THIS LOOK LIKE?
               text=row_text,
               metadata=metadata
            )

            # Show progress every 10 rows
            if i % 10 == 0:
               self.logger.info(f"  Processing row {i+1}...")

            doc.embedding = self.embedder.embed(row_text)
            documents.append(doc)
      
      return documents
   
   """
   Load entire text file as one document
   Returns single document in a list
   """
   def _load_text_file(self, filepath: Path) -> List[Document]:
      with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
         content = f.read()
   
      if not content.strip():
         return []
      
      metadata = self._extract_metadata(filepath)

      doc = Document(
         id=filepath.stem,
         text=content,
         metadata=metadata
      )

      doc.embedding = self.embedder.embed(content)

      return [doc]

   """ 
   Extract metadata from file and content
   FILE FORMAT:
   <SOURCE>_<DOMAIN>_<TYPE>_<YEAR>.{ext}
   SOURCE: Document origin, ex. "NHTSA"
   DOMAIN: Original document domain, ex. "transmission"
   TYPE:   Additional label, ex. "tsb
   YEAR:   Year of document creation, ex. "2026"
   """
   def _extract_metadata(
      self,
      filepath: Path,
      splitter: str='_'
   ) -> dict:
      metadata = {
         'filename': filepath.name,
         'path': str(filepath)
      }

      # Extract words in filename, split on spliter parameter
      parts = filepath.stem.split(splitter)

      # Additional metadata in file name (customize)
      if len(parts) == 4:
         metadata['source'] = parts[0]
         metadata['domain'] = parts[1]
         metadata['type']   = parts[2]
         metadata['year']   = parts[3]
      else:
         print(f"Warning: {filepath.name} doesn't match expected format")
         metadata['source'] = 'unknown'
         metadata['domain'] = 'unknown'

      return metadata

   """
   Load documents and ingest them into the graph
   Combines loading and ingestion into one step (may separate later)
   """
   def load_and_ingest(
      self,
      graph: KnowledgeGraph,
      extractor: QuestionExtractor,
      classifier: DomainClassifier,
      directory: str,
      domain: str='unknown',
      **kwargs
   ) -> dict:
      # Document statistics
      total_stats = {
         'docs_processed': 0,
         'questions_extracted': 0,
         'questions_added': 0,
         'duplicates': 0,
         'errors': 0
      }

      try:
         # Load documents
         documents = self.load_from_directory(directory, **kwargs)
      except Exception as e:
         self.logger.error(f"Failed to load documents: {e}", exc_info=True)
         return {'error': str(e)}
      
      if not documents:
         self.logger.warning("No documents loaded")
         return {
         'docs_processed': 0,
         'questions_extracted': 0,
         'questions_added': 0,
         'duplicates': 0,
         'errors': 0
         }
      
      # List of existing questions
      all_questions = []

      for i, doc in enumerate(documents, start=1):
         try:
            # Add document to graph
            graph.add_document(doc)

            # STEP ZERO: Identify domains for this document
            existing_domains = list(graph.domains.keys())
            doc_domains = classifier.identify_domain(doc.text, existing_domains)
            # Log domains
            self.logger.info(f"\nDoc {i}/{len(documents)}: {doc.id}")
            self.logger.info(f"  Identified domains: {', '.join(doc_domains)}")

            # STEP ONE: Check if this doc answers existing questions
            existing_answered = [] # TODO: CHANGE TO A SET FOR AUTO DEDUPING
            for domain in doc_domains:
               # Optimize, only check questions in the same domain
               relevant_questions = [
                  q for q in graph.questions.values()
                  if domain in q.domains
               ]
               # If no domain filter (early in ingestion), check all
               if not relevant_questions:
                  relevant_questions = list(graph.questions.values())

               # Check for relevant questions in domain
               for question in relevant_questions:
                  similarity = cosine_similarity(question.embedding, doc.embedding)
                  if similarity > 0.75: # Threshold
                     question.answers.source_documents.append(doc)
                     existing_answered.append(question.text)
            
            # STEP TWO: Extract new questions
            questions = extractor.extract_from_text(
               doc.text,
               doc,
               num_questions=5,
               existing_questions=all_questions + existing_answered # More context
            )

            # STEP 3: Add new questions to graph, update nearby nodes
            self.logger.info(f"\nProcessing doc [{i}/{len(documents)}]: {doc.metadata.get('filename', doc.id)}")
            for q_text, answers in questions:
               result = insert_question_smart(graph, q_text, answers, domain)
               # Successful question addition
               if result:
                  total_stats['questions_added'] += 1
                  all_questions.append(q_text) # Add to question list
                  self.logger.info(f"  Added: {q_text}")
               # Duplicate question, record failed addition
               else:
                  total_stats['duplicates'] += 1
                  self.logger.info(f"  Duplicate: {q_text}")
            
            # Update metadata
            total_stats['docs_processed'] += 1
            total_stats['questions_extracted'] += len(questions)

         # Record doc.id of failed node
         except Exception as e:
            self.logger.error(f"ERROR processing {doc.id}: {e}", exc_info=True)
            total_stats['errors'] += 1
            continue

      return total_stats
   
   """ Save document embeddings to output directory"""
   def save_embeddings(self, documents: List[Document], filename: str="embeddings.pkl") -> None:
      output_path = os.path.join(self.embeddings_dir, filename)

      # Save just the embeddings and IDs, less data
      embedding_data = {
         doc.id: {
            'embedding': doc.embedding,
            'metadata': doc.metadata
         }
         for doc in documents
      }

      # Drop data
      with open(output_path, 'wb') as f:
         pickle.dump(embedding_data, f)
      
      self.logger.info(f"Saved {len(embedding_data)} embeddings to {output_path}")

   """ Load saved document embeddings """
   def load_embeddings(self, filename: str="embeddings.pkl") -> dict:
      output_path = os.path.join(self.embeddings_dir, filename)

      if not os.path.exists(output_path):
         self.logger.warning(f"No saved embeddings found at {output_path}")
         return {}
      
      # Retrieve data
      with open(output_path, 'rb') as f:
         embedding_data = pickle.load(f)

      return embedding_data
   
   """Save extracted questions for a document"""
   def save_questions_cache(self, doc_id: str, questions: List[Tuple[str, List[Document]]]):
      cache_path = os.path.join(self.questions_dir, f"{doc_id}.json")
      
      # Convert to JSON-serializable format
      cache_data = [
         {
            'question': q_text,
            'answer_doc_ids': [doc.id for doc in answer_docs]
         }
         for q_text, answer_docs in questions
      ]
      
      with open(cache_path, 'w') as f:
         json.dump(cache_data, f, indent=2)
   
   """Load cached questions for a document"""
   def load_questions_cache(self, doc_id: str, all_documents: dict) -> Optional[List[Tuple[str, List[Document]]]]:
      cache_path = os.path.join(self.questions_dir, f"{doc_id}.json")
      
      if not os.path.exists(cache_path):
         return None
      
      try:
         with open(cache_path, 'r') as f:
            cache_data = json.load(f)
         
         # Reconstruct questions with Document objects
         questions = []
         for item in cache_data:
            q_text = item['question']
            answer_docs = [all_documents[doc_id] for doc_id in item['answer_doc_ids'] if doc_id in all_documents]
            questions.append((q_text, answer_docs))
      
         return questions
         
      except Exception as e:
         self.logger.warning(f"Failed to load question cache for {doc_id}: {e}")
         return None