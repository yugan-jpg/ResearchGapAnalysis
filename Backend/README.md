# Scientific Literature Gap Discovery backend

Phase 3 adds local MiniLM embeddings and MongoDB Atlas Vector Search to the existing FastAPI/PDF-ingestion foundation. It does **not** call an LLM or perform research-gap analysis.

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
│   ├── api/routes/embeddings.py   # Bounded, idempotent embedding backfill
│   ├── api/routes/search.py       # Semantic chunk retrieval
│   ├── db/mongodb.py              # Lazy PyMongo Atlas client
│   ├── db/indexes.py              # Initial collections/indexes
│   ├── modules/pdf_processing/    # Validation, extraction, chunking, storage
│   ├── modules/embeddings/        # Local encoder, backfill, vector repository
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
└── atlas_vector_index.json          # Atlas Vector Search index definition
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

## Phase 3 local embeddings and semantic retrieval

The backend uses `sentence-transformers/all-MiniLM-L6-v2` locally on CPU. Its vectors have **384 dimensions** and are normalized before storage. The first embedding/backfill request downloads model weights from Hugging Face into the local user cache; this can take a few minutes on a first run. A download/load failure is returned as a safe `503` response.

Phase 3 stores the following fields on each existing `paper_chunks` document without changing its PDF metadata: `embedding`, `embedding_model`, `embedding_dimensions`, `embedding_status`, and `embedded_at`.

### Configuration

The defaults in `.env.example` are suitable for this project:

```dotenv
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSIONS=384
EMBEDDING_BATCH_SIZE=16
EMBEDDING_DEVICE=cpu
VECTOR_SEARCH_INDEX_NAME=paper_chunks_vector_index
VECTOR_SEARCH_CANDIDATE_MULTIPLIER=10
```

On an 8 GB CPU-only computer, keep `EMBEDDING_DEVICE=cpu`. Reduce `EMBEDDING_BATCH_SIZE` to `8` if the first backfill puts too much pressure on memory.

### Atlas Vector Search setup (manual)

Atlas Vector Search indexes cannot be created by ordinary PyMongo collection-index commands. After running `python scripts/initialize_database.py`, create the vector index in Atlas:

1. Open your Atlas deployment, then select **Search** / **Atlas Search** and choose **Create Search Index**.
2. Choose the `MONGODB_DATABASE` database and the `paper_chunks` collection.
3. Select the JSON editor, name the index `paper_chunks_vector_index` (or set `VECTOR_SEARCH_INDEX_NAME` to your chosen name), and paste the contents of `atlas_vector_index.json`.
4. Create the index and wait until Atlas shows it as **Active** before calling semantic search.

The definition uses `embedding` as a 384-dimensional `cosine` vector and defines `paper_id` as a future filter field.

### Backfill existing or newly uploaded chunks

Start the API, then run an intentionally bounded backfill. The first request may download the model:

```powershell
$backfill = @{ limit = 50 } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/embeddings/backfill `
  -ContentType 'application/json' `
  -Body $backfill
```

Repeat until `remaining_chunks` is `0`. To embed one paper only, include its `paper_id`; use `force: true` only to deliberately regenerate an already-current embedding. The normal path skips chunks already marked `ready` with the same model and dimensions, making it safe to repeat after partial failures.

### Semantic search

After backfill is complete and the Atlas index is active:

```powershell
$query = @{ query = 'limitations of external validation in forecasting models'; top_k = 5 } | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/search/semantic `
  -ContentType 'application/json' `
  -Body $query
```

Results contain chunk text, parent `paper_id`, source page number, stable chunk index, and Atlas cosine similarity score. `top_k` accepts `1` through `50`; the API returns `503` with setup guidance if the vector index is absent or unavailable.

## Frontend connection

The existing frontend already reads `VITE_API_BASE_URL`. In `Frontend/.env`, set:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`. Phase 1 confirms the HTTP/CORS boundary through `/api/health`; paper and analysis routes remain deliberately unimplemented until their relevant phases.

## Initial MongoDB storage preparation

`python scripts/initialize_database.py` creates `papers`, `document_chunks`, `processing_jobs`, `claims`, `contradictions`, `research_gaps`, and `analysis_runs`, plus standard indexes for status, timestamps, IDs, DOI, hash, and chunk position. It does not create an Atlas Vector Search index yet because the Phase 2 embedding schema has not been defined.
