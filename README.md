# AI-Powered Scientific Literature Gap Discovery System

A research-intelligence application using Retrieval-Augmented Generation (RAG), MongoDB Atlas, and LLM-assisted scientific claim analysis. It compares evidence across papers and surfaces potential mismatch signals for further investigation. Research gap generation and validation remain future work.

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
- Cross-paper claim comparison using cosine similarity and shared scientific terms
- Trusted-claim selection and claim-aware candidate retrieval
- Scientific relationship classification, contradiction detection and independent model assessment
- Comparison queue with pagination, source references and duplicate pair prevention
- API usage monitoring, automatic throttling and verification retries
- Targeted re-verification and historical duplicate repair

## Setup

From the `Backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Configure `.env` with your MongoDB and Groq credentials.

Initialize MongoDB:

```powershell
python scripts/initialize_database.py
```

In MongoDB Atlas, create or update the `paper_chunks_vector_index` Vector Search index using [Backend/atlas_vector_index.json](Backend/atlas_vector_index.json). Wait until the index is ready. It uses 384-dimensional cosine vectors and filters for `paper_id`, `chunk_id`, `embedding_status`, `embedding_model`, and `embedding_dimensions`.

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

Optional comparison settings in `Backend/.env` control the minimum claim cosine, candidate limit and batch sizes:

```dotenv
COMPARISON_MIN_SIMILARITY=0.60
COMPARISON_NEIGHBORS=5
COMPARISON_DISCOVERY_LIMIT=5
COMPARISON_DETECTION_LIMIT=2
```

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
| POST | `/api/comparisons/similar` | Discover and queue cross-paper claim pairs |
| POST | `/api/comparisons/detect` | Classify queued pairs with the configured models |
| GET | `/api/comparisons` | List and filter stored comparisons |

## Testing

```powershell
python -m pytest -q
```

The optional live Groq smoke test is disabled by default.

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
Cross-Paper Candidate Retrieval & Similarity Screening
      ↓
Qwen and GPT-OSS — Independent Relationship Assessments
      ↓
Evidence-Linked Comparisons & Potential Mismatch Signals
      ↓
Research Gap Analysis (Planned)
```

Similarity and shared terms select candidates for review. Model assessments depend on the source context and are not proof of a scientific contradiction or research gap. See the [backend README](Backend/README.md) for the API workflow and limitations.

