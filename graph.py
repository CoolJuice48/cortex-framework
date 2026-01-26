# ------------------------------------------------------------------ #
# Cortex imports
from structs import Document, Question, are_same_question, cosine_similarity
from embed import Embedder
from classifier import DomainClassifier
# ------------------------------------------------------------------ #
# Python imports
from typing import Optional, List, Tuple
import numpy as np
import uuid
import json
# ------------------------------------------------------------------ #

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
   # Embed query
   query_embedding = graph.embedder.embed(question_text)
   
   # Compare to all questions
   results = []
   for question in graph.questions.values():
      similarity = cosine_similarity(query_embedding, question.embedding)
      results.append((question, similarity))
   
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
   
"""
Analyzes if a domain should split based on directional divergence
Returns a dict with shape
{
   bool: should_split,
   float: variance,
   float: avg_similarity,
   int: num_questions
}
"""
def analyze_domain_coherence(
      graph: KnowledgeGraph,
      domain_name: str='general'
) -> dict:
   questions = [q for q in graph.questions.values() if domain_name in q.domains]

   # Check: Minimum size to split
   if len(questions) < 10:
      return {
         'Should split': False,
         'Reason': 'insufficient_data'
      }

   # Get direction (centroids of answer document embeddings)
   directions = []
   for q in questions:
      if q.answer_documents:
         centroid = np.mean([doc.embedding for doc in q.answer_documents], axis=0)
         directions.append(centroid)
   
   if len(directions) < 2:
      return {
         'Should split': False,
         'Reason': 'insufficient_data'
      }
   
   # Compute variance in directions
   directions_array = np.array(directions)
   variance = np.var(directions_array, axis=0).mean()

   # Compute cosine similarity between all pairs
   similarities = []
   for i in range(len(directions)):
      for j in range(i+1, len(directions)):
         sim = cosine_similarity(directions[i], directions[j])
         similarities.append(sim)
   
   avg_similarity = np.mean(similarities)

   return {
      'should_split': variance > 0.15 or avg_similarity < 0.6, # Thresholds, must be tuned
      'variance': variance,
      'avg_similarity': avg_similarity,
      'num_questions': len(questions)
   }

"""
Uses an LLM to split domains into subdomains
"""
def split_domain(
   graph: KnowledgeGraph,
   classifier: DomainClassifier,
   domain_name: str="general"
) -> List[str]:
   questions = [q for q in graph.questions.values() if domain_name in q.domains]

   # Sample questions for LLM
   sample_questions = [q.text for q in questions[:20]]

   prompt = f"""The domain "{domain_name}" has become too broad and needs to split into more specific subdomains.

   Here are sample questions from this domain:
   {chr(10).join(f"- {q}" for q in sample_questions)}

   Based on these questions, suggest 2-4 subdomains to split this into.

   Requirements:
   - Subdomains should be specific and meaningful
   - Each subdomain should be clearly distinct
   - Use lowercase with underscores
   - Format as JSON array

   Example: ["automatic_transmission", "manual_transmission", "transmission_fluid"]

   Suggest subdomains now: (please and thank you!)"""

   response = classifier.client.messages.create(
      model="claude-sonnet-4-20250514",
      max_tokens=300,
      messages=[{"role": "user", "content": prompt}]
   )
   
   # Parse response
   try:
      text = response.content[0].text.strip()
      if "```json" in text:
         start = text.find("```json") + 7
         end = text.find("```", start)
         text = text[start:end].strip()
      elif "```" in text:
         parts = text.split("```")
         text = parts[1].strip()
      
      subdomains = json.loads(text)
      return subdomains
        
   except Exception as e:
      print(f"Error parsing subdomain response: {e}")
      return [f"{domain_name}_1", f"{domain_name}_2"]  # Fallback