# ------------------------------------------------------------------ #
# Cortex imports
from structs import Document, Question, Answer, are_same_question
from embed import Embedder, cosine_similarity
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
      self.questions = {} # question_id (str) -> Question
      self.documents = {} # document_id (str) -> Document
      self.domains = {}   # domain_name (str)  -> Domain
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
      answer: Answer,
      domain_names: List[str]=["general"],
      parent_id: Optional[str]=None
   ) -> Optional[Question]:
      # Create the question
      new_q = Question(
         text=text,
         answers=[answer],
         domains=domain_names
      )
      new_q.embedding = self.embedder.embed(text)

      # Check if question already exists (circular dependency)
      duplicate = self._find_duplicate(new_q)
      if duplicate:
         print(f"Duplicate question detected: '{text}' already exists as '{duplicate.text}'")
         return None
      
      # Generate unique ID
      q_id = new_q.id
      self.questions[q_id] = new_q

      # Link answer to this question
      answer.question_ids.add(q_id)

      # Parent specified: link question to parent
      if parent_id and parent_id in self.questions:
         parent = self.questions[parent_id]
         new_q.parents.append(parent)
         parent.children.append(new_q)

      # Add to domain(s)
      for domain_name in domain_names:
         # Create domain if doesn't exist
         if domain_name not in self.domains:
            from structs import Domain
            self.domains[domain_name] = Domain(
               name=domain_name,
               questions=set(),
               parent_domain=None
            )
         # Add question to domain
         domain = self.domains[domain_name]
         domain.questions.add(new_q)

         # If no parent, is root question
         if not parent_id:
            # Domain implicitly tracks roots
            pass
   
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
   def get_question_roots(self, domain_name: Optional[str]=None) -> List[Question]:
      if domain_name:
         if domain_name not in self.domains:
            return []
         domain = self.domains[domain_name]

         # Return only questions with no parents
         return [q for q in domain.questions if not q.parents]
      else:
         # Return all root questions accross all domains
         return [q for q in self.questions.values() if not q.parents]

   """ Returns root questions for a specified domain """
   def get_domain_roots(self, domain_name: str) -> List[Question]:
      return self.get_question_roots(domain_name)
   
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
Returns Question if added, none if duplicate
"""
def insert_question_smart(
   graph: KnowledgeGraph,
   text: str,
   answer: Answer,
   domain_names: List[str]=["general"]
) -> Optional[Question]:
   # Temp question to find parent
   temp_q = Question(
      text=text,
      answers=[answer],
      domains=domain_names
   )
   temp_q.embedding = graph.embedder.embed(text)

   # Check for duplicates
   if graph._find_duplicate(temp_q):
      return None
   
   # Find parent
   parent = find_parent_question(graph, temp_q)
   parent_id = parent.id if parent else None

   # Add to graph (add_question() handles domain creation)
   added_question = graph.add_question(
      text=text,
      answer=answer,
      domain_names=domain_names,
      parent_id=parent_id
   )

   return added_question

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
) -> Optional[List[Answer]]:
   results = query_graph(graph, question_text, top_k=1)
   
   if not results:
      return None  # Empty graph
   
   best_match, similarity = results[0]
   
   if similarity >= confidence_threshold:
      print(f"Cache hit! Similar to: '{best_match.text}' (similarity: {similarity:.3f})")
      return best_match.answers
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

   # Get domain object
   if domain_name not in graph.domains:
      return {
         'should_split': False,
         'reason': 'domain_not_found',
         'num_questions': 0
      }
   
   domain = graph.domains[domain_name]
   questions = list(domain.questions)

   # Check: Minimum size to split
   if len(questions) < 10:
      return {
         'should_split': False,
         'reason': 'insufficient_data',
         'num_questions': len(questions)
      }

   # Get direction (centroids of answer document embeddings)
   directions = []
   for q in questions:
      if q.answers:
         # Get all source documents from all answers
         all_docs = []
         for answer in q.answers:
            all_docs.extend(answer.source_documents)

         embs = [a.embedding for a in q.answers if a.embedding is not None]

         if not embs:
            # No embeddings available yet
            return {
               "domain": domain_name,
               "question_id": q.id,
               "status": "no_embeddings",
            }

         centroid = np.mean(np.asarray(embs, dtype=float), axis=0)

         if all_docs:
            centroid = np.mean([doc.embedding for doc in q.answers], axis=0)
            directions.append(centroid)
   
   if len(directions) < 2:
      return {
         'should_split': False,
         'reason': 'insufficient_data',
         'num_questions': len(questions)
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
Returns a list of new subdomain names
"""
def split_domain(
   graph: KnowledgeGraph,
   classifier: DomainClassifier,
   domain_name: str="general"
) -> List[str]:
   # Get domain object
   if domain_name not in graph.domains:
      print(f"  Domain '{domain_name}' not found")
      return []
   
   domain = graph.domains[domain_name]
   questions = list(domain.questions)

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
   
"""
Reassigns questions from old domain to new subdomains
Uses an LLM, but could use centroid + nearest centroid at risk of reduced accuracy
"""
def reassign_questions_to_subdomains(
   graph: KnowledgeGraph,
   classifier: DomainClassifier,
   old_domain_name: str,
   new_subdomain_names: List[str]
) -> None:
   # Get old domain
   if old_domain_name not in graph.domains:
      print(f"  Domain '{old_domain_name}' not found")
      return
   
   old_domain = graph.domains[old_domain_name]
   questions = list(old_domain.questions)

   print(f"\nReassigning {len(questions)} questions from '{old_domain_name}' to subdomains...")

   # Create new subdomain objects
   from structs import Domain
   for subdomain_name in new_subdomain_names:
      if subdomain_name not in graph.domains:
         new_subdomain = Domain(
            name=subdomain_name,
            questions=set(),
            parent_domain=old_domain # Link to parent
         )
         graph.domains[subdomain_name] = new_subdomain
         old_domain.subdomains.append(new_subdomain) # Track subdomain

   # Reassign each question
   for question in questions:
      prompt = f"""Which subdomain does this question belong to?

      Question: {question.text}

      Available subdomains:
      {chr(10).join(f"- {sd}" for sd in new_subdomain_names)}

      Return ONLY the subdomain name, nothing else."""

      # Query LLM
      response = classifier.client.messages.create(
         model="claude-sonnet-4-20250514",
         max_tokens=50,
         messages=[{"role": "user", "content": prompt}]
      )

      # Parse subdomain
      subdomain = response.content[0].text.strip().lower()

      # Validate subdomain
      if subdomain_name not in new_subdomain_names:
         subdomain_name = new_subdomain_names[0] # Fallback

      # Remove question from old domain
      old_domain.questions.discard(question) # Remove from set
      question.domains.remove(old_domain_name)

      # Add question to new subdomain
      new_subdomain = graph.domains[subdomain_name]
      new_subdomain.questions.add(question) # Add to set
      question.domains.append(subdomain_name)

      print(f"  '{question.text[:60]}...' → {subdomain}")

   # Keep old domain as parent, mark as split
   print(f"\nDomain '{old_domain_name}' split into {len(new_subdomain_names)} subdomains")

"""
Periodically check if any domains should split
Returns the number of domains that have been split
"""
def check_and_split_domains(
   graph: KnowledgeGraph,
   classifier: DomainClassifier,
   threshold_questions: int=50
) -> int:
   splits_performed = 0

   # Iterate over copy of keys
   for domain_name in list(graph.domains.keys()):
      domain = graph.domains[domain_name]
      
      # Skip if already split
      if domain.subdomains:
         continue

      # Skip if too small
      if len(domain.questions) < threshold_questions:
         continue

      # Analyze coherence
      analysis = analyze_domain_coherence(graph, domain_name)
      
      if analysis.get('should_split', False):
         print(f"\n{'='*60}")
         print(f"DOMAIN SPLIT TRIGGERED: {domain_name}")
         print(f"  Variance: {analysis.get('variance', 'N/A')}")
         print(f"  Avg similarity: {analysis.get('avg_similarity', 'N/A')}")
         print(f"  Questions: {analysis['num_questions']}")
         print(f"{'='*60}")
         
         # Split the domain
         new_subdomain_names = split_domain(graph, classifier, domain_name)
         print(f"New subdomains: {new_subdomain_names}")
         
         # Reassign questions
         reassign_questions_to_subdomains(
            graph,
            classifier,
            old_domain=domain_name,
            new_subdomains=new_subdomain_names
         )

         splits_performed += 1

   return splits_performed