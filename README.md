# Cortex Framework

Autonomous knowledge organization system using multiple LLM calls and hierarchical question graphs.

## Overview

The Cortex Framework builds a self-organizing knowledge graph from unstructured documents (tabular, papers, textbooks). Questions are extracted automatically, duplicates detected via circular dependency checking, and hierarchies emerge naturally from structural & semantic similarity.

## How to Navigate

Start by reading structs.py, which contains the Question, Answer, Domain, and Document dataclasses which the rest of the program is built on. See 'Project Structure' below to see which files they're used in.

## Installation
```bash
pip install -r requirements.txt
```

## Setup

1. Get Anthropic API key from https://console.anthropic.com
2. Set environment variable:
```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
```
3. Place your data in `./data/` directory

## Usage
```bash
python main.py
```

## Project Structure
```
Cortex_v1/
├── main.py              # Main execution script
├── structs.py           # Data structures (Document, Question)
├── embed.py             # Embedding functionality
├── graph.py             # Knowledge graph
├── extraction.py        # Question extraction & querying
├── loader.py            # Document loading
├── classifier.py        # Domain classification
├── data/                # Your data files (not committed)
├── log/                 # Log files (not committed)
└── output/              # Generated files (not committed)
```

## Core Concepts

- **Circular Dependency**: Two questions are duplicates if they point to the same underlying knowledge
- **Directional Divergence**: Domains split when questions point in divergent directions in embedding space
- **Hierarchical Self-Organization**: Parent-child relationships emerge from structural (question-answer pairs) & semantic similarity

## License

OptIn, L.L.C.
