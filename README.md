# AI-Powered Scientific Literature Gap Discovery System

A FastAPI-based scientific literature analysis system using Retrieval-Augmented Generation (RAG), MongoDB Atlas, and LLM-powered scientific claim extraction and verification.

## Tech Stack

- **Backend:** Python, FastAPI
- **Database:** MongoDB Atlas, PyMongo
- **PDF Processing:** PyMuPDF
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search:** MongoDB Atlas Vector Search
- **Claim Extraction:** Groq (GPT-OSS-120B)
- **Claim Verification:** Google Gemini Flash
- **Frontend:** React, Vite

## Features

- Scientific PDF upload, text extraction, and chunking
- Local embedding generation (384 dimensions)
- Semantic search using MongoDB Atlas Vector Search
- LLM-powered scientific claim extraction
- Source evidence validation and duplicate detection
- Secondary Gemini verification for scientific claims
- Numerical consistency and claim completeness checks
- MongoDB storage for papers, embeddings, and claims

## Setup

From the `Backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure `.env` with your MongoDB, Groq, and Gemini API credentials.

Initialize MongoDB:

```powershell
python scripts/initialize_database.py
```

Start the backend:

```powershell
uvicorn app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

## Environment Configuration

```dotenv
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL_NAME=openai/gpt-oss-120b
LLM_API_KEY=your_groq_api_key

VERIFIER_ENABLED=true
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-3.8-flash
```

Additional MongoDB, embedding, and verification settings are documented in `.env.example`.

Never commit your actual `.env` file.

## API Endpoints

| Method | Endpoint                         | Description               |
| ------ | -------------------------------- | ------------------------- |
| GET    | `/api/health`                    | Backend health check      |
| POST   | `/api/papers/upload`             | Upload scientific PDF     |
| GET    | `/api/papers`                    | List uploaded papers      |
| POST   | `/api/embeddings/backfill`       | Generate embeddings       |
| POST   | `/api/search/semantic`           | Semantic search           |
| POST   | `/api/claims/extract/{paper_id}` | Extract scientific claims |
| POST   | `/api/claims/verify/{paper_id}`  | Verify existing claims    |
| GET    | `/api/claims/{paper_id}`         | Retrieve extracted claims |

## Testing

```powershell
python -m pytest -q
```

## RAG Workflow

```text
Scientific PDFs
      ↓
Text Extraction & Chunking
      ↓
Embedding Generation
      ↓
MongoDB Atlas Vector Search
      ↓
Groq — Scientific Claim Extraction
      ↓
Rule-Based Evidence Validation
      ↓
Gemini — Secondary Verification
      ↓
Verified / Review / Rejected Claims
      ↓
Research Gap Analysis (Planned)
```

## Upcoming Features

- Cross-paper claim comparison
- Contradiction detection
- Research gap identification
- Feasibility assessment
- Research gap ranking

