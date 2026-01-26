#!/bin/bash
timestamp=$(date +"%Y-%m-%d %H:%M")
git add .
git commit -m "Update: $timestamp"
git push
echo "Pushed with timestamp: $timestamp"