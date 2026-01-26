from embed import cosine_similarity
from dataclasses import dataclass
from typing import List, Optional, Set
import numpy as np
import uuid

@dataclass
class Document:
   id: str                                # uuid4(), used to find what question an answer references
   text: str                              # Raw text of document
   embedding: Optional[np.ndarray]=None   # Embedded document
   metadata: dict=None                    # Source, date, author, etc.

@dataclass
class Question:
   # TODO: INCLUDE ANSWERS
   id: str                                # uuid4(), used to find what question an answer references
   text: str                              # Raw text of question
   embedding: Optional[np.ndarray]=None   # Embedded question
   answer_documents: List[Document]=None  # Documents which answer the question TODO: CHANGE TO answers: List[Answer]=None
   children: List['Question']=None        # List of follow-up questions
   parents: List['Question']=None         # What this question is a follow-up to
   neighbors: List['Question']=None       # NEW: Similar questions not directly related (uses answer_documents as a check)
   domains: List[str]=None                # Which domains this question falls under
   confidence: float=1.0                  # How confident the model is in the validity of the question

   def __post_init__(self):
      if self.answer_documents is None:
         self.answer_documents = []       # TODO: Set implementation?
      if self.children is None:
         self.children = []               # TODO: Set implementation?
      if self.parents is None:
         self.parents = []                # TODO: Set implementation?
      if self.domains is None:
         self.domains = []                # TODO: Set implementation?

@dataclass
class Answer:
   question_id: List[str]                 # TODO: Must point to host question(s) IDs
   text: str                              # Raw text of question
   embedding: Optional[np.ndarray]=None   # Embedded answer
   answer_documents: List[Document]=None  # List of documents which support this answer
   confidence: float=1.0                  # How confident the model is in the accuracy of the answer

   def __post_init__(self):
      if self.answer_documents is None:
         self.answer_documents = []       # TODO: Set implementation?

@dataclass
class Domain:
   id: str                                # uuid4(), used to map new questions to existing domains
   name: str='general'                    # Domain name
   questions: Set[Question]               # Set of questions, inherently deduplicates

   def __init__(self, name: str, questions: Set[Question]):
      self.id=uuid.uuid4()                # Create new ID
      self.name = name                    # Name is as provided
      self.questions = questions          # Questions are as provided (as a set)

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