# Phase 1 — Backend foundation

This lightweight FastAPI foundation is intentionally limited to configuration, MongoDB Atlas connectivity, initial collection/index preparation, CORS, and a health route. It does **not** process PDFs, call an LLM, generate embeddings, run vector search, or perform research-gap analysis.

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
│   ├── db/mongodb.py              # Lazy PyMongo Atlas client
│   ├── db/indexes.py              # Initial collections/indexes
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

## Frontend connection

The existing frontend already reads `VITE_API_BASE_URL`. In `Frontend/.env`, set:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`. Phase 1 confirms the HTTP/CORS boundary through `/api/health`; paper and analysis routes remain deliberately unimplemented until their relevant phases.

## Initial MongoDB storage preparation

`python scripts/initialize_database.py` creates `papers`, `document_chunks`, `processing_jobs`, `claims`, `contradictions`, `research_gaps`, and `analysis_runs`, plus standard indexes for status, timestamps, IDs, DOI, hash, and chunk position. It does not create an Atlas Vector Search index yet because the Phase 2 embedding schema has not been defined.
