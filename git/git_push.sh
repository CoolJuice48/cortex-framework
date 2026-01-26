#!/bin/bash

# ============================================
# EDIT YOUR COMMIT MESSAGE HERE:
# ============================================
commit_message="Switched to all-MiniLM-L6-v2 for embeddings for faster testing"
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