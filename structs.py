# ------------------------------------------------------------------ #
# Cortex imports
from embed import cosine_similarity
# ------------------------------------------------------------------ #
# Python imports
from dataclasses import dataclass, field
from typing import List, Optional, Set
import numpy as np
import uuid
# ------------------------------------------------------------------ #

"""
A document, either a line of tabular data or full text file
"""
@dataclass
class Document:
   id: str                                # uuid4(), used to find what question an answer references
   text: str                              # Raw text of document
   embedding: Optional[np.ndarray]=None   # Embedded document
   metadata: dict=None                    # Source, date, author, etc.

""" An answer to one or more questions, supported by a document """
@dataclass
class Answer:
   id: str=field(default_factory=lambda: str(uuid.uuid4()))
   text: str=""                                                  # Raw text of question
   embedding: Optional[np.ndarray]=None                          # Embedded answer
   source_documents: List[Document]=field(default_factory=list)  # List of documents which support this answer
   confidence: float=1.0                                         # How confident the model is in the accuracy of the answer

   # Which questions does this answer?
   question_ids: Set[str] = field(default_factory=set)           # Use set for deduplication

""" An LLM-generated question about a certain document """
@dataclass
class Question:
   id: str=field(default_factory=lambda: str(uuid.uuid4()))    # uuid4(), used to find what question an answer references
   text: str=""                                                # Raw text of question
   embedding:    Optional[np.ndarray]=None                     # Embedded question
   answers:      List[Answer]=field(default_factory=list)      # Answers to the question
   children:     List['Question']=field(default_factory=list)  # List of follow-up questions
   parents:      List['Question']=field(default_factory=list)  # What this question is a follow-up to
   neighbors:    Set['Question']=field(default_factory=set)    # Similar questions not directly related (uses answer_documents as a check)
   domains:      List[str]=field(default_factory=list)         # Which domains this question falls under (names as strings)
   confidence:   float=1.0                                     # How confident the model is in the validity of the question

   def __hash__(self):
      return hash(self.id)

"""
A domain of knowledge
Graph { Domain -> Set[Question] -> List[Answer] -> List[Document] }
"""
@dataclass
class Domain:
   id: str=field(default_factory=lambda: str(uuid.uuid4()))     # uuid4(), used to map new questions to existing domains
   name: str='general'                                          # Domain name
   questions: Set[Question]=field(default_factory=set)          # Set of questions, deduplicates
   
   parent_domain: List['Domain'] = field(default_factory=list)  # Parent domain(s), may be multiple
   subdomains: List['Domain']=field(default_factory=list)       # Subdomain(s)

   centroid: Optional[np.ndarray]=None                          # Avg. embedding, for splitting
   variance: float=0.0                                          # Spread of questions

"""
Checks if two questions point to the same source knowledge
Returns True if they do, False if they don't
"""
def has_circular_dependency(
   q1: Question,
   q2: Question,
   threshold: float=0.5
) -> bool:
   if not q1.answers or not q2.answers:
      return False
   
   # Get all source documents from all answers
   q1_doc_ids = set()
   for answer in q1.answers:
      q1_doc_ids.update(doc.id for doc in answer.source_documents)

   q2_doc_ids = set()
   for answer in q2.answers:
      q2_doc_ids.update(doc.id for doc in answer.source_documents)

   if not q1_doc_ids or not q2_doc_ids:
      return False

   # Do they point to the exact same documents?
   if q1_doc_ids == q2_doc_ids:
      return True
   
   # Do answer docs have similar embeddings?
   q1_doc_embeddings = []
   for answer in q1.answers:
      for doc in answer.source_documents:
         if doc.embedding is not None:
            q1_doc_embeddings.append(doc.embedding)

   q2_doc_embeddings = []
   for answer in q2.answers:
      for doc in answer.source_documents:
         if doc.embedding is not None:
            q2_doc_embeddings.append(doc.embedding)

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

"""
Finds the centroid of a domain in embedding space (axis for divergence checking)
Returns centroid as a numpy array, or None if no embeddings
"""
def find_centroid(domain: Domain) -> Optional[np.ndarray]:
   if not domain.questions:
      return None
   
   embeddings = [q.embedding for q in domain.questions if q.embedding is not None]

   if not embeddings:
      return None
   
   return np.mean(embeddings, axis=0)

"""
Do two questions share similar source documents?
Determines if they belong to the same subdomain
Returns True if they share enough documents (Jaccard similarity > threshold)
"""
def share_docs(q1: Question, q2: Question, threshold: float=0.5) -> bool:
   # Get source documents from all answers
   q1_docs = set()
   for answer in q1.answers:
      q1_docs.update(doc.id for doc in answer.source_documents)

   q2_docs = set()
   for answer in q2.answers:
      q2_docs.update(doc.id for doc in answer.source_documents)

   if not q1_docs or not q2_docs:
      return False
   
   # Jaccard similarity: intersection / union
   intersection = len(q1_docs & q2_docs)
   union = len(q1_docs | q2_docs)

   overlap = intersection / union if union > 0 else 0
   return overlap > threshold

"""
Find similar questions that aren't parents or children
These are candidates for domain clustering
"""
def find_neighbors(
   question: Question,
   all_questions: List[Question],
   similarity_threshold: float=0.7,
   max_neighbors: int=10
) -> Set[Question]:
   neighbors = set()

   # Get existing parents and children to exclude
   exclude = set(question.parents + question.children + [question])

   for other in all_questions:
      if other in exclude:
         continue

      # Check similarity
      sim = cosine_similarity(question.embedding, other.embedding)
      if sim > similarity_threshold:
         neighbors.add(other)

      if len(neighbors) >= max_neighbors:
         break
   
   return neighbors

"""
Recursively find subdomain clusters within a domain
Returns a list of new subdomains discovered
"""
def check_divergence(
   domain: Domain,
   all_questions: List[Question],
   classifier,
   depth: int = 0,
   max_depth: int=3,
   min_divergence: float=0.3,
   min_cluster_size: int=5
) -> List[Domain]:
   # Base case: too deep or too few questions
   if depth >= max_depth or len(domain.questions) < min_cluster_size * 2:
      return []
   
   # Find domain axis (centroid)
   axis = find_centroid(domain)
   if axis is None:
      return []
   
   # Prepare for new subdomains, track already processed
   new_subdomains = []
   processed = set()

   # Search for divergent questions
   for question in domain.questions:
      if question in processed:
         continue
      
      # Check alignment with domain axis
      alignment = cosine_similarity(question.embedding, axis)

      # If divergent enough, check for cluster
      if alignment < (1.0 - min_divergence):  # Low alignment = high divergence
         # Build cluster around divergent question
         cluster = {question}
         processed.add(question)

         # Process neighbors and check if they share docs (indicates shared subdomain)
         if not question.neighbors:
            question.neighbors = find_neighbors(question, list(domain.questions))

         for neighbor in question.neighbors:
            if neighbor in processed:
               continue

            # Check for shared source documents (same subdomain)
            if share_docs(question, neighbor, threshold=0.5):
               cluster.add(neighbor)
               processed.add(neighbor)

         # If cluster is satisfactory, create subdomain
         if len(cluster) >= min_cluster_size:
            # Use LLM to name subdomain
            sample_questions = [q.text for q in list(cluster)[:15]]
            subdomain_name = classifier.name_subdomain(sample_questions, domain.name)

            new_subdomain = Domain(
               name=subdomain_name,
               questions=cluster,
               parent_domain=domain
            )
            new_subdomain.centroid = find_centroid(new_subdomain)

            new_subdomains.append(new_subdomain)
            domain.subdomains.append(new_subdomain)

            # Recursive check on this subdomain
            deeper_subdomains = check_divergence(
               new_subdomain,
               all_questions,
               classifier,
               depth=depth + 1,
               max_depth=max_depth,
               min_divergence=min_divergence,
               min_cluster_size=min_cluster_size
            )
            new_subdomains.extend(deeper_subdomains)

   return new_subdomains