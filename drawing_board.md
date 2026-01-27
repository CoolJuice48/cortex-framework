## 01/26/26
TODO:
- Global set of IDs for domains, docs, etc? To help speed data retrieval & catalogging

DONE:
- Determine where to use sets (anywhere duplication may occur, ask if it's better)
- Refactor graph to involve domains, which contain questions, which contain answers
- Change functions to involve Domain, Answer, and modified Question classes
- Questions are contained in a domain, which holds a set of questions for deduplication
   - Allows for the same question to appear in multiple domains
- Each Answer contains a list of question_IDs, as each answer may answer multiple questions

"""
Checks whether a node's domain should be split into subdomains
Recursively hunts for clusters within a domain. 'split' param records successful divergence
TODO: Determine where to stop (depth <= max_depth)
Returns a list containing the new subdomain(s), which contain the reassigned questions
"""
fn check_divergence(domain: Domain, depth: int=0, split: bool=False, min_sim: float=0.1) -> List[Domain]:
   place_at_origin(domain) # Place domain cluster at (0, 0, ..., 0) Probably not needed
   axis = find_centroid(domain, axis=0) # Find vector which best describes D, use as base for comparison (SVD?)
   new_domains = []

   for question in domain.questions:
      # Angle is too small to consider for splitting
      # Rationale: Over large distances, new domains diverge as a separate cluster from middle_vec
      if cosine_similarity(question.embedding, axis) < min_sim:
         continue
      
      # The document is different enough that, approaching infinity, a new cluster emerges near it
      # Set this as the new axis, look for similar documents (indicates they share this new domain with axis)
      else:
         new_question_cluster = set() # Set to remove potential duplicates
         root = question
         for neighbor in question.neighbors: # NEW MEMBER: neighbors are similar questions that might not be in the same domain
            # Do they share data? (Gear subdomain of Transmission has different questions, but same answer_docs)
            ## THIS MUST BE PERFECT -- the quality of new domains relies on this
            ## share_docs() checks the semantic & structural similarity of answer documents
            if share_docs(neighbor, root): # Maybe 50% of both parameters answer_documents as a minimum?
               new_question_cluster.add(neighbor)
            else:
               continue # Maybe each question has a "last_checked" param, if not checked for a while trigger a divergence check?
         
         # Create new domain from diverged documents' questions
         if new_question_cluster:
            new_domain_name = name_domain(new_question_cluster)
            new_domain = Domain(new_domain_name, new_question_cluster) # __init__ creates ID
            new_domains.append(new_domain)

            DOMAIN_GRAPH.add_subdomain(new_domain) # Add new domain to graph, graph helper function takes care of data reorganization
            DOMAIN_IDs.append(new_domain.id) # Add domain ID to set, helps catalog domains
            
            # RECURSION ON NEIGHBOR, NEW CLUSTER FOUND
            check_divergence(neighbor.domain, depth+=1, split=True)

         else:
            # RECURSION ON NEIGHBOR, NO NEW CLUSTER FOUND
            check_divergence(neighbor.domain, depth+=1, split=False) # Don't want to go O(n^3) with list of domains
   
   return new_domains

## 01/27/26
TODO:
- Determine best Q&A linking structure
- Refactor KnowledgeGraph to be lighter & hashable
   - Graph should ONLY store ids, NOT the entire objects

DONE:
-

"""
Intake query as string
Returns:
   query as Question object if is_new_question()
   Otherwise returns existing question equal to query
"""
def intake_query(query: str) -> Question:
   embedded_query = self.embedder.embed(query)

   if is_new_question(embedded_query):
      question = Question(
         text=query,
         embedding=embedded_query
      )
      graph.add_question(question)
      return Question

   else:
      return find_existing_question(embedded_query)

"""
Checks if embedded query is an existing question
Returns:
   True if existing, False if not
"""
def is_new_question(question: np.ndarray) -> bool:
   similar_qs = find_similar_questions(question)
   for q in similar_qs:
      if are_same_question(question, q.embedding):
         return True
      else:
         return False

def can_be_answered_now(question: np.ndarray) -> bool:
