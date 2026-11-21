# Scientific Literature Gap Discovery — Backend

Python FastAPI backend for processing scientific research papers, generating embeddings, performing semantic search, and extracting and verifying scientific claims using LLMs.

## Technologies

- **FastAPI** — REST API
- **MongoDB Atlas / PyMongo** — Database
- **PyMuPDF** — PDF processing
- **Sentence Transformers** — Local embeddings
- **MongoDB Atlas Vector Search** — Semantic retrieval
- **Groq (GPT-OSS-120B)** — Claim extraction
- **Google Gemini Flash** — Secondary claim verification

## Project Structure

```text
Backend/
├── app/
│   ├── main.py
│   ├── api/routes/
│   ├── core/config.py
│   ├── db/
│   ├── modules/
│   │   ├── pdf_processing/
│   │   ├── embeddings/
│   │   ├── claim_extraction/
│   │   └── ...
│   └── schemas/
├── scripts/
│   └── initialize_database.py
├── tests/
├── atlas_vector_index.json
├── requirements.txt
└── .env.example
```

## Installation

Open PowerShell inside the `Backend` directory.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Configure your `.env` before starting the backend.

## Environment Variables

```dotenv
# MongoDB
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=your_database_name

# Embeddings
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=16

# Vector Search
VECTOR_SEARCH_INDEX_NAME=paper_chunks_vector_index

# Groq
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_API_KEY=your_groq_api_key
LLM_MODEL_NAME=openai/gpt-oss-120b

# Gemini
VERIFIER_ENABLED=true
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL_NAME=gemini-3.8-flash
```

Other configuration options are available in `.env.example`.

**Never commit `.env` or expose API keys.**

## MongoDB Setup

Initialize the database and standard indexes:

```powershell
python scripts/initialize_database.py
```

In MongoDB Atlas, create a Vector Search index on the `paper_chunks` collection using `atlas_vector_index.json`.

Index name: `paper_chunks_vector_index`

## Running the Backend

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

- **API:** http://127.0.0.1:8000
- **Swagger Documentation:** http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Function |
|---|---|---|
| GET | `/api/health` | Check backend and database |
| POST | `/api/papers/upload` | Upload and process a PDF |
| GET | `/api/papers` | List uploaded papers |
| GET | `/api/papers/{paper_id}` | Retrieve paper metadata |
| POST | `/api/embeddings/backfill` | Generate chunk embeddings |
| POST | `/api/search/semantic` | Search research paper chunks |
| POST | `/api/claims/extract/{paper_id}` | Extract scientific claims |
| POST | `/api/claims/verify/{paper_id}` | Verify existing claims |
| GET | `/api/claims/{paper_id}` | Retrieve stored claims |

## Backend Features

### PDF Processing

- Upload text-based PDFs up to 20 MB
- Extract and divide text into overlapping chunks
- Store paper metadata and chunks in MongoDB
- Detect duplicate uploads using SHA-256

### Embeddings and Semantic Search

- Generate 384-dimensional embeddings locally using `all-MiniLM-L6-v2`
- Store embeddings in MongoDB Atlas
- Retrieve relevant chunks through Atlas Vector Search
- Support incremental embedding generation

### Scientific Claim Extraction

- Extract structured claims using Groq's `openai/gpt-oss-120b`
- Validate supporting evidence against source text
- Store claims with paper IDs, chunk IDs, and page numbers
- Prevent duplicate claim records

### Gemini Claim Verification

- Perform secondary verification using Gemini
- Check evidence support, numerical consistency, completeness, and claim categories
- Flag possible duplicate or incomplete claims
- Assign `accepted`, `review`, or `rejected` verdicts
- Preserve claims when verification is unavailable
- Support verification of previously extracted claims without re-extraction

## Testing

Run backend tests:

```powershell
python -m pytest -q
```

Use Swagger to test PDF uploads, semantic search, claim extraction, and verification.

## Current Status

| Component | Status |
|---|---|
| PDF ingestion | Implemented |
| MongoDB storage | Implemented |
| Embedding generation | Implemented |
| Semantic search | Implemented |
| Groq claim extraction | Implemented and tested |
| Gemini verification | Implemented; live testing pending |
| Contradiction detection | Planned |
| Research gap analysis | Planned |




