#!/bin/bash

# ============================================
# EDIT COMMIT MESSAGE HERE:
# ============================================
commit_message="Major restructuring
- Changed Answer, Question, and Domain classes to store ids, not full objects
- Changed KnowledgeGraph to store Question IDs, no longer Documents
- Focused on making data storage hashable & fast
- Improved Document metadata, pulls domain info from document name"
# ============================================

echo "Pulling latest changes first..."
git pull

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