#!/bin/bash

# ============================================
# EDIT COMMIT MESSAGE HERE:
# ============================================
commit_message="Add Answer and Domain struct implementations in functions"
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