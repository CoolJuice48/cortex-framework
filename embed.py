from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
   def __init__(self, model_name='all-mpnet-base-v2'):
      """
      all-MiniLM-L6-v2: Fast, 384 dimensions, good quality
      all-mpnet-base-v2: Slower, 768 dimensions, better quality
      """
      self.model = SentenceTransformer(model_name)
      self.dimension = self.model.get_sentence_embedding_dimension()

   def embed(self, text: str) -> np.ndarray:
      return self.model.encode(text, convert_to_numpy=True)
   
   def embed_batch(self, texts: List[str]) -> np.ndarray:
      return self.model.encode(texts, convert_to_numpy=True)

"""
Calculates cosine similarity between two vectors
Returns: similarity_score
Used in: find_most_similar()
"""
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
   dot_product = np.dot(a, b)
   norm_a = np.linalg.norm(a)
   norm_b = np.linalg.norm(b)
   return dot_product / (norm_a * norm_b)

"""
Finds most similar embedding from a list
Returns: (index, similarity_score)
"""
def find_most_similar(query_embedding: np.ndarray, candidates: List[np.ndarray]) -> tuple[int, float]:
   similarities = [cosine_similarity(query_embedding, c) for c in candidates]
   best_idx = np.argmax(similarities)
   return best_idx, similarities[best_idx]