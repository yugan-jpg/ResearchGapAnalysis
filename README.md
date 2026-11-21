AI-Powered
&#x20;     ↓
Text Ex Scientific Literature Gap Discovery System
==================================================

A FastAPI-based backend for a scientific literature analysis system using Retrieval-Augmented Generation (RAG), MongoDB Atlas, and semantic vector search.

## Tech Stack

- **Backend:** Python, FastAPI
- **Database:** MongoDB Atlas, PyMongo
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** MongoDB Atlas Vector Search
- **LLM:** Qwen3-235B-A22B via OpenRouter (planned)
- **Frontend:** React, Vite

## Features

- MongoDB Atlas integration
- Scientific PDF processing and text chunking
- Embedding generation (384 dimensions)
- Semantic search using MongoDB Atlas Vector Search
- REST API endpoints for embeddings and search
- Research gap detection and LLM-based analysis (planned)

## Setup

From the `Backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure your `.env` with your MongoDB connection string and other required settings.

Initialize the database:

```powershell
python scripts/initialize_database.py
```

Start the backend:

```powershell
uvicorn app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

## Testing

```powershell
python -m pytest tests/test_embeddings.py -v
```

## RAG Workflow

```text
Scientific PDFstraction & Chunking
      ↓
Embedding Generation
      ↓
MongoDB Atlas Vector Search
      ↓
Relevant Research Passages
      ↓
LLM Analysis (Planned)
      ↓
Research Gap Identification (Planned)
```

## Upcoming Features

- LLM integration for research question answering
- Scientific claim extraction
- Contradiction detection
- Research gap analysis and ranking
- Feasibility assessment
