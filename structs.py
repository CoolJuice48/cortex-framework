from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from embed import cosine_similarity

@dataclass
class Document:
   id: str
   text: str
   embedding: Optional[np.ndarray] = None
   metadata: dict = None # Source, date, author, etc.

@dataclass
class Question:
   text: str
   embedding: Optional[np.ndarray] = None
   answer_documents: List[Document] = None
   children: List['Question'] = None
   parents: List['Question'] = None
   domains: List[str] = None
   confidence: float = 1.0

   def __post_init__(self):
      if self.answer_documents is None:
         self.answer_documents = []
      if self.children is None:
         self.children = []
      if self.parents is None:
         self.parents = []
      if self.domains is None:
         self.domains = []

"""
Checks if two questions point to the same source knowledge
Returns: True if they do, False if they don't
"""
def has_circular_dependency(q1: Question, q2: Question, threshold: float=0.5) -> bool:
   if not q1.answer_documents or not q2.answer_documents:
      return False
   
   # Do they point to the same documents?
   q1_doc_ids = {doc.id for doc in q1.answer_documents}
   q2_doc_ids = {doc.id for doc in q2.answer_documents}
   if q1_doc_ids == q2_doc_ids:
      return True
   
   # Do answer docs have similar embeddings?
   q1_doc_embeddings = [doc.embedding for doc in q1.answer_documents if doc.embedding is not None]
   q2_doc_embeddings = [doc.embedding for doc in q2.answer_documents if doc.embedding is not None]
   if not q1_doc_embeddings or not q2_doc_embeddings:
      return False
   
   # Compare centroids of answer documents
   q1_centroid = np.mean(q1_doc_embeddings, axis=0)
   q2_centroid = np.mean(q2_doc_embeddings, axis=0)
   similarity = cosine_similarity(q1_centroid, q2_centroid)
   return similarity > threshold

"""
Checks semantic similarity and circular dependency
Returns: True if same question, False if not
"""
def are_same_question(q1: Question, q2: Question, threshold: float=0.7) -> bool:
   # Check 1: Are they similar?
   similarity = cosine_similarity(q1.embedding, q2.embedding)
   if similarity < threshold:
      return False
   
   # Check 2: Do they point to the same knowledge?
   return has_circular_dependency(q1, q2)