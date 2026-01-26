from typing import Optional, List, Tuple
import uuid
from structs import Document, Question, are_same_question, cosine_similarity
from embed import Embedder

class KnowledgeGraph:
   def __init__(self, embedder: Embedder):
      self.questions = {} # id -> Question
      self.documents = {} # id -> Document
      self.domains = {}   # domain_name -> list of root questions
      self.embedder = embedder

   """ Adds a document to the graph """
   def add_document(self, doc: Document) -> None:
      # Embed document text if needed
      if doc.embedding is None:
         doc.embedding = self.embedder.embed(doc.text)

      # Add document to graph at doc.id
      self.documents[doc.id] = doc

   """
   Adds a question to the graph
   Returns None if there's a duplicate (circular dependency)
   Returns the question if it's new or needs to be added
   """
   def add_question(
      self,
      text: str,
      answer_documents: List[Document],
      domain: str="general",
      parent_id: Optional[str]=None
   ) -> Optional[Question]:
      # Create the question
      new_q = Question(
         text=text,
         answer_documents=answer_documents,
         domains=[domain]
      )
      new_q.embedding = self.embedder.embed(text)

      # Check if question already exists (circular dependency)
      duplicate = self._find_duplicate(new_q)
      if duplicate:
         print(f"Duplicate question detected: '{text}' already exists as '{duplicate.text}'")
         return None
      
      # Generate unique ID
      q_id = str(uuid.uuid4())
      self.questions[q_id] = new_q

      # Parent specified: link question to parent
      if parent_id and parent_id in self.questions:
         parent = self.questions[parent_id]
         new_q.parents.append(parent)
         parent.children.append(new_q)
      # No parent specified: root question for domain
      else:
         if domain not in self.domains:
            self.domains[domain] = []
         self.domains[domain].append(new_q)

      return new_q
   
   """
   Check if question already in graph
   Uses circular dependency check
   """
   def _find_duplicate(self, new_q: Question) -> Optional[Question]:
      for existing_q in self.questions.values():
         if are_same_question(new_q, existing_q):
            return existing_q
      return None

   """ Returns all questions with no parents """
   def get_question_roots(self, domain: Optional[str]=None) -> List[Question]:
      if domain:
         return self.domains.get(domain, [])
      else:
         return [q for q in self.questions.values() if not q.parents]

   """ Returns root questions for a specified domain """
   def get_domain_roots(self, domain: str) -> List[Question]:
      return self.domains.get(domain, [])
   
"""
Find the most appropriate parent for a new question
Returns the most similar existing question that's more general
"""
def find_parent_question(
   graph: KnowledgeGraph,
   new_q: Question,
   threshold: float=0.6
) -> Optional[Question]:
   # Empty graph, no parent
   if not graph.questions:
      return None
   
   candidates = []

   for q in graph.questions.values():
      similarity = cosine_similarity(new_q.embedding, q.embedding)
      if similarity > threshold:
         candidates.append((q, similarity))

   # No parent found, new root
   if not candidates:
      return None
   
   # Return most similar candidate
   candidates.sort(key=lambda x: x[1], reverse=True)
   return candidates[0][0]

"""
Insert a question and automatically find its parent
"""
def insert_question_smart(
   graph: KnowledgeGraph,
   text: str,
   answer_documents: List[Document],
   domain: str="general"
) -> Optional[Question]:
   # Temp question to find parent
   temp_q = Question(text=text, answer_documents=answer_documents)
   temp_q.embedding = graph.embedder.embed(text)

   # Check for duplicates
   if graph._find_duplicate(temp_q):
      return None
   
   # Find parent
   parent = find_parent_question(graph, temp_q)

   # Add to graph
   q_id = str(uuid.uuid4())
   graph.questions[q_id] = temp_q
   temp_q.domains = [domain]

   if parent:
      temp_q.parents.append(parent)
      parent.children.append(temp_q)
   else:
      if domain not in graph.domains:
         graph.domains[domain] = []
      graph.domains[domain].append(temp_q)

   return temp_q

"""
Finds the most relevant questions in the graph.
Returns list of (Question, similarity_score) tuples.
"""
def query_graph(
   graph: KnowledgeGraph,
   question_text: str,
   top_k: int = 5
) -> List[Tuple[Question, float]]:
   query_embedding = graph.embedder.embed(question_text)
   
   results = []
   for q in graph.questions.values():
      similarity = cosine_similarity(query_embedding, q.embedding)
      results.append((q, similarity))
   
   # Sort by similarity, return top k
   results.sort(key=lambda x: x[1], reverse=True)
   return results[:top_k]

"""
Answers a question by finding matching cached Q&A.
Returns answer documents if found, None if need to retrieve.
"""
def answer_question(
   graph: KnowledgeGraph,
   question_text: str,
   confidence_threshold: float = 0.8
) -> Optional[List[Document]]:
   results = query_graph(graph, question_text, top_k=1)
   
   if not results:
      return None  # Empty graph
   
   best_match, similarity = results[0]
   
   if similarity >= confidence_threshold:
      print(f"Cache hit! Similar to: '{best_match.text}' (similarity: {similarity:.3f})")
      return best_match.answer_documents
   else:
      print(f"No cached answer (best match: {similarity:.3f})")
      return None