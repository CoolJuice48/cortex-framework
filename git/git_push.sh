#!/bin/bash

# ============================================
# EDIT YOUR COMMIT MESSAGE HERE:
# ============================================
commit_message="NO_MESSAGE"
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