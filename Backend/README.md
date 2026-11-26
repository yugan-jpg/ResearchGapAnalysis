# Scientific Literature Gap Discovery — Backend

FastAPI backend for scientific PDF processing, semantic search, claim extraction and verification using RAG.

## Tech Stack

- Python, FastAPI
- MongoDB Atlas, PyMongo
- PyMuPDF
- Sentence Transformers (`all-MiniLM-L6-v2`)
- MongoDB Atlas Vector Search
- Groq — Qwen 3.8 27B (Extraction)
- Groq — GPT-OSS-120B (Verification)

## Features

- Layout-aware PDF extraction and chunking
- Local embeddings and semantic search
- Scientific claim extraction and independent verification
- Evidence, numerical and completeness validation
- Duplicate detection and canonical claim selection
- API monitoring, throttling and retries
- Source provenance and targeted re-verification

## Setup

From the `Backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` with MongoDB and Groq credentials. Do not overwrite an existing `.env`.

```dotenv
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=research_gap_db

LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_API_KEY=your_groq_api_key
LLM_MODEL_NAME=qwen/qwen3.8-27b

VERIFIER_ENABLED=true
VERIFIER_MODEL_NAME=openai/gpt-oss-120b
```

Additional settings are available in `.env.example`.

Initialize MongoDB:

```powershell
python scripts/initialize_database.py
```

Create the Atlas Vector Search index `paper_chunks_vector_index` on the `paper_chunks` collection using `atlas_vector_index.json`.

Never commit `.env` or API credentials.

## Running the Backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Swagger: http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint |
|---|---|
| GET | `/api/health` |
| POST | `/api/papers/upload` |
| GET | `/api/papers` |
| GET | `/api/papers/{paper_id}` |
| POST | `/api/embeddings/backfill` |
| POST | `/api/search/semantic` |
| POST | `/api/claims/extract/{paper_id}` |
| POST | `/api/claims/verify/{paper_id}` |
| GET | `/api/claims/{paper_id}` |

Maintenance scripts for historical repair, duplicate resolution and targeted re-verification are available in `scripts/`. Use `--help` for usage.

## Testing

```powershell
python -m pytest -q
```

Latest results: **110 passed, 1 skipped** (optional live Groq test).

Model verdicts are assessments, not proof of scientific correctness.



