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
      
      prompt = f"""Analyze this document and identify what domain(s) it belongs to.

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