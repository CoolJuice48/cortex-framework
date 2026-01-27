# ------------------------------------------------------------------ #
# Cortex imports
from structs import Document, Answer
from graph import KnowledgeGraph, insert_question_smart
from embed import Embedder
# ------------------------------------------------------------------ #
# Python imports
import anthropic
import json
from typing import List, Tuple
# ------------------------------------------------------------------ #

class QuestionExtractor:
   def __init__(self, api_key: str, embedder: Embedder):
      self.client = anthropic.Anthropic(api_key=api_key)
      self.embedder = embedder

   """
   Uses an LLM to generate questions a given document answers
   Returns list of (question_text, Answer) tuples
   """
   def extract_from_text(
      self,
      text: str,
      doc: Document,
      num_questions: int=5,
      existing_questions: List[str]=None
   ) -> List[Tuple[str, Answer]]:
      # Check running list of existing questions
      existing_q_text = ""
      if existing_questions and len(existing_questions) > 0:
         # Only show top 50 recent questions to avoid huge prompts
         recent = existing_questions[-50:]
         existing_q_text = f"""
         IMPORTANT: Do NOT generate questions similar to these already-asked questions:
         {chr(10).join(f"- {q}" for q in recent)}

         Your questions must be DIFFERENT and ask about NEW aspects of the document.
         """
               
      prompt = f"""Given this document, generate {num_questions} questions that this document can answer.

      Document:
      {text[:2000]}

      {existing_q_text}

      Requirements:
      - Questions should be at different levels of specificity (broad to specific)
      - Questions should be directly answerable from the document
      - Questions must be UNIQUE and not overlap with existing questions above
      - For each question, provide a brief answer (1-2 sentences) based on the document
      - Format as a JSON array of objects with "question" and "answer" fields

      Example format:
      [
      {{"question": "What is X?", "answer": "X is a component that..."}},
      {{"question": "How does Y work?", "answer": "Y works by..."}}
      ]

      Generate the questions and answers now:"""

      response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500, # Increased to include answers
            messages=[{"role": "user", "content": prompt}]
      )

      # Parse response
      try:
         response_text = response.content[0].text.strip()

         # Remove markdown fences
         if "```json" in response_text:
            # Extract content between ```json and ```
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
         elif "```" in response_text:
            # Handle plain ``` fences
            parts = response_text.split("```")
            response_text = parts[1].strip() if len(parts) > 1 else parts[0]

         qa_pairs = json.loads(response_text.strip())

         # Create Answer objects
         results = []
         for item in qa_pairs:
            question_text = item.get('question', '')
            answer_text = item.get('answer', '')

            if not question_text:
               continue

            # Create answer object
            answer = Answer(
               text=answer_text,
               source_documents=[doc],
               confidence=1.0
            )

            if answer_text:
               answer.embedding = self.embedder.embed(answer_text)
            
            results.append((question_text, answer))

         return results

      # Parsing error
      except Exception as e:
         print(f"Error parsing LLM response: {e}")
         print(f"Raw response: {response.content[0].text}")
         return []

"""
Ingest a document: extract questions with LLM and add to graph
Returns stats dict with shape
{
extracted,
added,
duplicate_counts
}
"""
def ingest_document_with_extractor(
   graph: KnowledgeGraph,
   doc: Document,
   extractor: QuestionExtractor,
   domain_names: List[str]=["general"],
   existing_questions: List[str]=None
) -> dict:
   # Add document to graph
   graph.add_document(doc)

   # Extract questions using LLM
   print(f"Extracting questions from document: {doc.id}")
   qa_pairs = extractor.extract_from_text(
      doc.text,
      doc,
      num_questions=5,
      existing_questions=existing_questions
   )

   stats = {
      "extracted": len(qa_pairs),
      "added": 0,
      "duplicates": 0
   }

   for q_text, answer in qa_pairs:
      # Attempt to add question (rejected if duplicate)
      result = insert_question_smart(
         graph,       # KnowledgeGraph
         q_text,      # Question as text
         answer,      # Pass answer obkect
         domain_names # List of domain names
      )
      
      if result is not None:
         stats["added"] += 1
         print(f"  Added '{q_text}'")
      else:
         stats["duplicates"] += 1
         print(f"  Duplicate '{q_text}'")

   return stats