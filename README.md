# Lexical Search Engine

## Overview

A lexical search engine built with FastAPI that indexes text documents using manual preprocessing, TF-IDF vectorization, and cosine similarity ranking.

This project was implemented for the Spritle Gen AI backend task and intentionally avoids pretrained or external NLP tooling.

## Constraints

This project follows the assignment constraints:

- No pretrained embeddings
- No external NLP or vectorization libraries
- Manual TF-IDF implementation
- Manual cosine similarity implementation

## Features

- `GET /search?q=<query>`
  - Returns the top 3 most relevant documents
  - Includes `document`, cosine similarity `score`, and `snippet`
- `GET /index`
  - Rebuilds the in-memory index from the `documents/` folder
- Modular code for preprocessing, vectorization, indexing, ranking, and API handling
- Unit tests for all core components
- minimal HTML UI served from `/`

## Approach

The system follows a lexical retrieval pipeline:

1. Load `.txt` documents from the `documents/` folder.
2. Preprocess text using lowercasing, regex-based punctuation cleanup, tokenization, and stopword removal.
3. Build a shared vocabulary across all documents.
4. Compute TF-IDF vectors manually for each document.
5. Precompute vector norms for cosine similarity scoring.
6. At query time, preprocess and vectorize the query with the same vocabulary and IDF values.
7. Compute cosine similarity against each document and return the top 3 ranked matches.

This implementation is intentionally simple and manual to satisfy the assignment constraints.

## Implementation Details

- **Preprocessing:** lowercasing, punctuation removal via regex, tokenization, stopword filtering
- **Vectorization:** manual vocabulary construction, TF, DF, IDF, and TF-IDF generation
- **Ranking:** cosine similarity computed manually using Euclidean norms
- **Indexing:** documents loaded into an in-memory index; vectors and norms are precomputed

## Project Structure

```text
lexical-search-engine/
├── app/
│   ├── api.py
│   ├── indexer.py
│   ├── models.py
│   ├── preprocess.py
│   ├── search.py
│   ├── vectorizer.py
│   └── static/
│       └── index.html
├── documents/
├── tests/
│   ├── test_api.py
│   ├── test_indexer.py
│   ├── test_preprocess.py
│   ├── test_search.py
│   └── test_vectorizer.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running The Project

Start the server:

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --reload
```

Server URL: `http://127.0.0.1:8000`

## Sample API Calls

Search:

```bash
curl "http://127.0.0.1:8000/search?q=artificial%20intelligence%20in%20finance"
```

Rebuild index:

```bash
curl "http://127.0.0.1:8000/index"
```

## Example Response

`GET /search?q=artificial%20intelligence%20in%20finance`

```json
{
  "results": [
    {
      "document": "doc_02_ai_in_finance.txt",
      "score": 0.3041210964279587,
      "snippet": "**AI in Finance: Transforming the Landscape of Financial Services** The integration of artificial intelligence (AI) into the finance sector is revolutionizing..."
    },
    {
      "document": "doc_38_ai_and_ethics.txt",
      "score": 0.09234999282880407,
      "snippet": "**AI and Ethics: Navigating the Challenges of Tomorrow** As artificial intelligence (AI) continues to advance at an unprecedented pace..."
    },
    {
      "document": "doc_22_ai_and_data_privacy.txt",
      "score": 0.09118889918817716,
      "snippet": "**AI and Data Privacy: Navigating the Intersection of Innovation and Protection** Artificial Intelligence (AI) has emerged as a transformative force..."
    }
  ]
}
```

## Run Tests

```bash
pytest -q
```

Last verified locally on 2026-03-26: `29 passed, 14 warnings in 1.01s`

## Assumptions

- Only `.txt` files in `documents/` are indexed
- The index is stored in memory
- Query terms missing from the vocabulary are ignored

## Limitations

- This is a lexical search system, not semantic search
- Ranking depends on word overlap after preprocessing
- Synonyms and deeper semantic relationships are not modeled
- In-memory indexing is not optimized for very large corpora
- Snippets are simple preview text, not contextual highlights

## Design Choices

Raw term frequency and smoothed IDF are used to keep the implementation simple, transparent, and easy to explain. The project prioritizes correctness, modularity, and clarity over advanced optimizations.
