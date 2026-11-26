# AI-Powered Scientific Literature Gap Discovery System

A scientific literature analysis system using Retrieval-Augmented Generation (RAG), MongoDB Atlas, and LLM-powered scientific claim extraction and verification to identify research gaps.

## Tech Stack

- **Backend:** Python, FastAPI
- **Database:** MongoDB Atlas, PyMongo
- **PDF Processing:** PyMuPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** MongoDB Atlas Vector Search
- **Claim Extraction:** Groq — Qwen 3.8 27B
- **Claim Verification:** Groq — GPT-OSS-120B
- **Frontend:** React, Vite

## Features

- Scientific PDF upload and layout-aware text extraction
- Text chunking and local embedding generation (384 dimensions)
- Semantic search using MongoDB Atlas Vector Search
- LLM-powered scientific claim extraction
- Independent factual verification and source evidence validation
- Numerical consistency and claim completeness checks
- Exact and semantic duplicate detection
- Canonical claim selection and incomplete paraphrase resolution
- Source provenance and verification audit history
- API usage monitoring, automatic throttling and verification retries
- Targeted re-verification and historical duplicate repair

## Setup

From the `Backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` with your MongoDB and Groq credentials.

Initialize MongoDB:

```powershell
python scripts/initialize_database.py
```

Start the backend:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

From the `Frontend` directory:

```powershell
npm install
npm run dev
```

## Environment Configuration

```dotenv
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL_NAME=qwen/qwen3.8-27b
LLM_API_KEY=your_groq_api_key

VERIFIER_ENABLED=true
VERIFIER_MODEL_NAME=openai/gpt-oss-120b
```

Both models use the same Groq API credentials.

Additional MongoDB, embedding, verification and rate-limit settings are documented in `.env.example`.

Never commit your actual `.env` file.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health check |
| POST | `/api/papers/upload` | Upload scientific PDF |
| GET | `/api/papers` | List uploaded papers |
| GET | `/api/papers/{paper_id}` | Retrieve paper details |
| POST | `/api/embeddings/backfill` | Generate embeddings |
| POST | `/api/search/semantic` | Semantic search |
| POST | `/api/claims/extract/{paper_id}` | Extract and verify scientific claims |
| POST | `/api/claims/verify/{paper_id}` | Verify existing claims |
| GET | `/api/claims/{paper_id}` | Retrieve extracted claims |

## Testing

```powershell
python -m pytest -q
```

Latest regression test results:

- 110 passed
- 1 skipped (optional live Groq test)

## RAG Workflow

```text
Scientific PDFs
      ↓
Layout-Aware Text Extraction & Chunking
      ↓
Embedding Generation
      ↓
MongoDB Atlas Storage & Vector Search
      ↓
Qwen — Scientific Claim Extraction
      ↓
Rule-Based Evidence & Numerical Validation
      ↓
GPT-OSS — Independent Factual Verification
      ↓
Duplicate Resolution & Canonical Selection
      ↓
Verified Scientific Claims
      ↓
Research Gap Analysis (Planned)
```




