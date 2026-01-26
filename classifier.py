# ------------------------------------------------------------------ #
# Cortex imports
from structs import Document
# ------------------------------------------------------------------ #
# Python imports
import anthropic
import json
from typing import List
# ------------------------------------------------------------------ #

class DomainClassifier:
   def __init__(self, api_key: str):
      self.client = anthropic.Anthropic(api_key=api_key)

   """
   Asks LLM to identify which domain(s) a document belongs to
   Returns list of domains (can be multiple)
   """
   def identify_domain(self, text: str, existing_domains: List[str]=None) -> List[str]:
      existing_text = ""
      if existing_domains:
         existing_text = f"""
         Existing domains in the system:
         {chr(10).join(f"- {d}" for d in existing_domains)}

         You can assign to existing domains OR suggest new ones if this content doesn't fit.
         """
      
      prompt = f"""Analyze this/these document(s) and identify what domain(s) it/they belong(s) to.

      Document:
      {text[:1500]}

      {existing_text}

      Requirements:
      - Use 1-3 specific domains (e.g., "transmission", "brake_system", "engine_cooling")
      - Be specific but not overly narrow
      - Use existing domains when appropriate
      - Suggest new domains when content is distinct
      - Format as JSON array of lowercase strings with underscores

      Example: ["transmission", "power_train"]

      Identify the domains now (thank you!):"""

      response = self.client.messages.create(
         model="claude-sonnet-4-20250514",
         max_tokens=200,
         messages=[{"role": "user", "content": prompt}]
      )

      # Parse response
      try:
         domains_text = response.content[0].text.strip()
         
         # Handle markdown fences
         if "```json" in domains_text:
               start = domains_text.find("```json") + 7
               end = domains_text.find("```", start)
               domains_text = domains_text[start:end].strip()
         elif "```" in domains_text:
               parts = domains_text.split("```")
               domains_text = parts[1].strip()
         
         domains = json.loads(domains_text)
         return domains
         
      except Exception as e:
         print(f"Error parsing domain response: {e}")
         return ["general"]  # Fallback
      
   """
   Classifies multiple documents at once for better domain consistency
   Returns a dict mapping document_id -> list of domains
   """
   def classify_document_batch(
   self,
   documents: List[Document],
   existing_domains: List[str] = None,
   batch_size: int = 10
   ) -> dict:
      results = {}
      
      # Process in batches to avoid token limits
      for i in range(0, len(documents), batch_size):
         batch = documents[i:i + batch_size]
         
         # Build batch prompt
         doc_summaries = []
         for j, doc in enumerate(batch, start=1):
            preview = doc.text[:500].replace('\n', ' ')  # Short preview
            doc_summaries.append(f"{j}. {doc.id}: {preview}...")
         
         existing_text = ""
         if existing_domains:
            existing_text = f"""
            Existing domains in the system:
            {chr(10).join(f"- {d}" for d in existing_domains)}

            You should prefer existing domains when documents fit, but suggest new ones when needed.
            """
            
         prompt = f"""Analyze these documents and identify what domain(s) each belongs to.

         {chr(10).join(doc_summaries)}

         {existing_text}

         Requirements:
         - Assign 1-3 specific domains per document
         - Be consistent across related documents
         - Use lowercase with underscores (e.g., "transmission", "brake_system")
         - Format as JSON object mapping document number to domain array

         Example:
         {{
         "1": ["transmission", "power_train"],
         "2": ["brake_system"],
         "3": ["engine_cooling", "electrical_system"]
         }}

         Classify all documents now:"""

         response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
         )
         
         # Parse response
         try:
            text = response.content[0].text.strip()
            
            # Handle markdown fences
            if "```json" in text:
               start = text.find("```json") + 7
               end = text.find("```", start)
               text = text[start:end].strip()
            elif "```" in text:
               parts = text.split("```")
               text = parts[1].strip()
            
            classifications = json.loads(text)
            
            # Map back to document IDs
            for j, doc in enumerate(batch, start=1):
               doc_key = str(j)
               if doc_key in classifications:
                  results[doc.id] = classifications[doc_key]
               else:
                  results[doc.id] = ["general"]  # Fallback
         
         except Exception as e:
            print(f"Error parsing batch classification: {e}")
            # Fallback: assign general to all
            for doc in batch:
               results[doc.id] = ["general"]
      
      return results