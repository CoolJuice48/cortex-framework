#!/bin/bash

# ============================================
# EDIT YOUR COMMIT MESSAGE HERE:
# ============================================
commit_message="Domain identification & divergence added
NEW:
ITEM: class DomainClassifier, FILE: classifier.py
- Uses an LLM API call to identify the domain of a given document

ITEM: analyze_domain_coherence(), FILE: graph.py
- Identifies when a domain should break into subdomains, based on cosine similarity and variance

ITEM: split_domain(), FILE: graph.py
- Generates subdomain names using another LLM API call

ITEM: reassign_questions_to_subdomains(), FILE: graph.py
- Domain reassignment using a third LLM API call, could use centroid and nearest centroid instead
ITEM: check_and_split_domains(), FILE: graph.py
- For use during large data ingestion, splits domains & reorganizes when needed"
# ============================================

echo "Adding all changes..."
git add .

echo "Committing with message:"
echo "---"
echo "$commit_message"
echo "---"

git commit -m "$commit_message"

echo "Pushing to GitHub..."
git push

echo "Done."