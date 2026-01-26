from structs import Document
from graph import KnowledgeGraph, insert_question_smart
import anthropic
import json
from typing import List, Tuple

class QuestionExtractor:
   def __init__(self, api_key: str):
      self.client = anthropic.Anthropic(api_key=api_key)

   """
   Uses an LLM to generate questions a given document answers
   Returns list of (question_text, [source_document]) tuples
   """
   def extract_from_text(
      self,
      text: str,
      doc: Document,
      num_questions: int=5,
      existing_questions: List[str]=None
   ) -> List[Tuple[str, List[Document]]]:
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
      - Format as a JSON array of strings
      - Don't include questions the document doesn't answer

      Example format:
      ["What is X?", "How does Y work?", "What causes Z?"]

      Generate the questions now:"""

      response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
      )

      # Parse response
      try:
         questions_text = response.content[0].text
         # Extract JSON array
         questions_text = questions_text.strip()

         # Remove markdown fences
         if "```json" in questions_text:
            # Extract content between ```json and ```
            start = questions_text.find("```json") + 7
            end = questions_text.find("```", start)
            questions_text = questions_text[start:end].strip()
         elif "```" in questions_text:
            # Handle plain ``` fences
            parts = questions_text.split("```")
            questions_text = parts[1].strip() if len(parts) > 1 else parts[0]

         questions = json.loads(questions_text.strip())

         # Return as (question, [doc]) tuples
         return [(q, [doc]) for q in questions]
      
      # Parsing error
      except Exception as e:
         print(f"Error parsing LLM response: {e}")
         print(f"Raw response: {response.content[0].text}")
         return []

"""
Ingest a document: extract questions with LLM and add to graph
Returns stats about what was added
"""
def ingest_document_with_extractor(
   graph: KnowledgeGraph,
   doc: Document,
   extractor: QuestionExtractor,
   domain: str="general",
   existing_questions: List[str]=None
) -> dict:
   # Add document to graph
   graph.add_document(doc)

   # Extract questions using LLM
   print(f"Extracting questions from document: {doc.id}")
   questions = extractor.extract_from_text(
      doc.text,
      doc,
      num_questions=5,
      existing_questions=existing_questions
   )

   stats = {
      "extracted": len(questions),
      "added": 0,
      "duplicates": 0
   }

   for q_text, answer_docs in questions:
      # Attempt to add question (rejected if duplicate)
      result = insert_question_smart(graph, q_text, answer_docs, domain)
      if result is not None:
         stats["added"] += 1
         print(f"  Added '{q_text}'")
      else:
         stats["duplicates"] += 1
         print(f"  Duplicate '{q_text}'")

   return stats