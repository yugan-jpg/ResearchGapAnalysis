# Phase 1 — Backend foundation

Phase 2 adds text-based scientific PDF ingestion to the existing FastAPI foundation. It validates one upload at a time, extracts per-page text with PyMuPDF, creates page-aware text chunks, and stores metadata and chunks in MongoDB Atlas. It does **not** call an LLM, generate embeddings, run vector search, or perform research-gap analysis.

## Structure

```text
Backend/
├── .env.example
├── .gitignore
├── requirements.txt
├── app/
│   ├── main.py                    # FastAPI app, lifespan, CORS
│   ├── core/config.py             # Environment variables
│   ├── api/routes/health.py       # GET /api/health
│   ├── api/routes/papers.py       # PDF upload, paper lookup, paper list
│   ├── db/mongodb.py              # Lazy PyMongo Atlas client
│   ├── db/indexes.py              # Initial collections/indexes
│   ├── modules/pdf_processing/    # Validation, extraction, chunking, storage
│   ├── schemas/health.py
│   └── modules/                   # Empty boundaries for later phases
│       ├── pdf_processing/
│       ├── claim_extraction/
│       ├── structural_similarity/
│       ├── contradiction_detection/
│       ├── research_gap_analysis/{empirical,methodological,theoretical,application}/
│       ├── feasibility_checker/
│       ├── devils_advocate/
│       └── research_gap_ranking/
└── scripts/initialize_database.py
└── tests/                          # Isolated PyMuPDF/mongomock tests
```

## Configuration

Copy `.env.example` to `.env` and enter your Atlas URI. `.env` is ignored by Git. `LLM_API_KEY` and `LLM_MODEL_NAME` are configuration placeholders for later phases only; no Phase 1 code reads the key for any external request.

## Launch with PowerShell

From the `Backend` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env in VS Code and replace only the placeholder values.
python scripts/initialize_database.py
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, run this once for the current shell only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## Test Phase 1

With the API running in one PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
Invoke-WebRequest http://127.0.0.1:8000/docs
Invoke-WebRequest -Method Options -Uri http://127.0.0.1:8000/api/health -Headers @{ Origin = 'http://localhost:5173'; 'Access-Control-Request-Method' = 'GET' }
```

A configured Atlas instance returns `status: ok` and `database.connected: true`. A missing or unreachable Atlas configuration returns HTTP 503 with a safe, credential-free explanation.

## Phase 2 PDF ingestion

`POST /api/papers/upload` accepts exactly one `file` form field. The backend streams it to a temporary file in bounded chunks, caps it at 20 MB, checks the filename, MIME type, `%PDF-` signature, PDF structure, password status, and extractable text. Temporary source files are deleted after processing.

The duplicate policy is SHA-256 based: uploading an already processed PDF returns HTTP `409 Conflict` and the existing `paper_id`; no duplicate chunks are inserted. Failed MongoDB writes are compensated by deleting the incomplete paper/chunks where possible, and no `processed` status is written before chunk insertion succeeds.

Text chunks default to 1,200 characters with a 200-character overlap, never cross page boundaries, and retain 1-based PDF page numbers. Configure these values with `PDF_CHUNK_SIZE` and `PDF_CHUNK_OVERLAP`; overlap must be smaller than chunk size.

Test a PDF through the interactive API page after starting the server: open `http://127.0.0.1:8000/docs`, expand **POST /api/papers/upload**, select a text-based PDF smaller than 20 MB, and execute. Use `GET /api/papers/{paper_id}` with the returned ID to retrieve stored metadata. `GET /api/papers` lists recent uploads.

To inspect Atlas after a successful upload, open the database named by `MONGODB_DATABASE`; inspect the `papers` and `paper_chunks` collections. Run `python scripts/initialize_database.py` once after configuring Atlas to create the indexes, including the unique SHA-256 and `(paper_id, chunk_index)` indexes.

Run the local tests with:

```powershell
python -m pytest -q
```

## Frontend connection

The existing frontend already reads `VITE_API_BASE_URL`. In `Frontend/.env`, set:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`. Phase 1 confirms the HTTP/CORS boundary through `/api/health`; paper and analysis routes remain deliberately unimplemented until their relevant phases.

## Initial MongoDB storage preparation

`python scripts/initialize_database.py` creates `papers`, `document_chunks`, `processing_jobs`, `claims`, `contradictions`, `research_gaps`, and `analysis_runs`, plus standard indexes for status, timestamps, IDs, DOI, hash, and chunk position. It does not create an Atlas Vector Search index yet because the Phase 2 embedding schema has not been defined.
