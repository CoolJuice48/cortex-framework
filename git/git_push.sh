#!/bin/bash

# ============================================
# EDIT COMMIT MESSAGE HERE:
# ============================================
commit_message="Update structs.py, now includes Answer and Domain classes
- Answer contains a pointer to a list of parent questions, allows for "root" answers
- Domain contains a set of questions, unique ID, and name as a string
- Planned directional divergence implementation in drawing_board.md
- HAVE NOT restructured existing class calls at this time"
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