# Scientific Literature Gap Discovery — Backend

Python FastAPI backend for processing scientific research papers, generating embeddings, performing semantic search, and extracting and verifying scientific claims using LLMs.

## Technologies

- **FastAPI** — REST API
- **MongoDB Atlas / PyMongo** — Database
- **PyMuPDF** — PDF processing
- **Sentence Transformers** — Local embeddings
- **MongoDB Atlas Vector Search** — Semantic retrieval
- **Groq (Qwen 3.8 27B)** — Claim extraction
- **Groq (GPT-OSS-120B)** — Secondary claim verification

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
LLM_MODEL_NAME=qwen/qwen3.8-27b
VERIFIER_ENABLED=true
VERIFIER_MODEL_NAME=openai/gpt-oss-120b
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2
VERIFIER_TIMEOUT_SECONDS=60
VERIFIER_MAX_RETRIES=2
VERIFIER_BATCH_SIZE=3
```

Other configuration options are available in `.env.example`.

**Never commit `.env` or expose API keys.**

Update an existing `.env` manually; do not replace it with `.env.example` or
lose your MongoDB and Groq credentials. Both models use the same URL and key.
Confirm the account is on Groq's free plan and both models are allowed on its
limits/model-permissions pages before making a live request. The application
cannot enforce the account's billing plan. Rate limits vary by account; 429
responses are retried a bounded number of times, then surfaced or marked
unavailable. Groq's model list API can be used to check account availability.
Existing Gemini-verified records are unchanged; use the existing verification
endpoint with `force: true` to explicitly re-evaluate them with Groq.

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

- Extract structured claims using Groq's `qwen/qwen3.8-27b`
- Validate supporting evidence against source text
- Store claims with paper IDs, chunk IDs, and page numbers
- Prevent duplicate claim records

### Groq Claim Verification

- Perform secondary verification using Groq's `openai/gpt-oss-120b`
- Check evidence support, numerical consistency, completeness, and claim categories
- Flag possible duplicate or incomplete claims
- Assign `accepted`, `review`, or `rejected` verdicts
- Preserve claims when verification is unavailable
- Support verification of previously extracted claims without re-extraction

Extraction skips chunks already marked complete, even when the primary model
changes. Use `force: true` only for deliberate re-extraction. New exact matches
within a paper are skipped and their source references are recorded; historical
duplicate documents are not merged. Run `python scripts/initialize_database.py`
to create the sparse unique `research_claim_normalized_unique` index before
concurrent extraction. Existing claims are not migrated automatically.

Extraction response `accepted_claims` is the historical count of newly stored
claims, not the number accepted by verification. `duplicate_claims` counts
exact records skipped. The additional `generated_claims`, `stored_claims`,
`extraction_rejections`, `exact_duplicates_skipped`, duplicate flag counts,
and `verification_*` counters report distinct stages. For existing claims,
verification can be retried via `/api/claims/verify/{paper_id}` without another
extraction request. Numerical source warnings require human review and do not
assert that sample groups are disjoint. `verification_accepted` is the number
accepted by the verifier in the extraction response; it can be lower than
`accepted_claims`.

For a previously ingested paper with incorrect layout order, run the following
from `Backend`. The first command previews a separate linked paper version;
the second creates it. The PDF must match the original stored SHA-256.

```powershell
python scripts/reingest_layout_version.py --paper-id "<original-paper-id>" --pdf "<absolute-path-to-original.pdf>"
python scripts/reingest_layout_version.py --paper-id "<original-paper-id>" --pdf "<absolute-path-to-original.pdf>" --apply
```

The old records stay intact. The new version has fresh chunks, pending
embeddings, and no inherited claims. Run `/api/embeddings/backfill` and
`/api/claims/extract/{paper_id}` explicitly for the new paper ID if needed.
Versions are the same paper, not independent corroborating publications. The
PDF itself was not retained by ingestion and must be supplied again.

For source-reference backfill, preview with
`python scripts/backfill_claim_sources.py --paper-id "<original-paper-id>"`,
then repeat with `--apply`. Only existing exact passages with matching paper,
chunk, and page are used. To re-evaluate historical duplicate candidates, use
`POST /api/claims/verify/{paper_id}` with `force: true`; this contacts Groq.

Inspect historical duplicate links without model calls using
`python scripts/repair_claim_duplicates.py --paper-id "<paper-id>"`. Add
`--apply` to record the planned metadata changes and their prior values.
Identical normalized claims can be linked to an eligible verified canonical
claim; uncertain relationships are unlinked and require re-verification.
Repairs that require another check set `reverification_required`; normal
verification (`force: false`) processes these claims without selecting other
review claims. Successful checks clear the marker; failed checks retain it.
For repairs made before this marker existed, preview recovery with
`python scripts/recover_reverification_queue.py --paper-id "<paper-id>"` and
add `--apply` to queue only matching stale repairs. Previous verification
results remain in `verification_history`.

Groq monitoring logs per-model local requests and reported/estimated tokens,
plus quota/reset headers when returned. Optional per-model TPM/RPM budgets,
a safety fraction, output-token estimate, maximum wait, and malformed
verification retries are documented in `.env.example`. Zero budgets disable
local limits. Monitor state is per backend process and does not coordinate
other processes or account activity.

Model verdicts are assessments, not proof of scientific correctness; evaluate
them against manually reviewed claims. Supporting passages are checked against
the original chunk, and model and source changes invalidate cached results.

Factual verdicts and duplicate assessments are stored separately. Confirmed,
source-checked equivalents can resolve to one canonical claim without an
additional model request; supported partial overlaps retain their unique
information. Review existing decisions with
`python scripts/resolve_claim_duplicates.py --paper-id "<paper-id>"` and add
`--apply` to apply the audited plan. The dry run does not call an LLM. Older
review records may lack the original factual verdict or duplicate candidate ID;
their missing decisions are not inferred from explanation text.
If a legacy claim needs a fresh check, preview
`python scripts/queue_claim_reverification.py --paper-id "<paper-id>" --claim-id "<claim-id>"`
(repeat `--claim-id` as needed), then add `--apply`. This queues only the
specified source-checked review claims and makes no model request itself;
normal verification processes them later.

## Testing

Run backend tests:

```powershell
python -m pytest -q
```

`tests/fixtures/claim_quality_labels.json` contains a small synthetic set with
manually assigned development labels. Record model predictions in the format
described in `scripts/score_claim_quality.py` and run that script to calculate
separate extraction, verification, and duplicate metrics. The fixture is not
an independently reviewed scientific benchmark.

For an optional live smoke test, confirm your Groq free-plan limits first and
set `RUN_GROQ_QUALITY_SMOKE=1` in PowerShell, then run
`python -m pytest -q tests/test_claim_live_smoke.py`. This checks model access
and sends at most one extraction and one verification request, excluding
bounded retries. It is a smoke check, not a quality evaluation.

Use Swagger to test PDF uploads, semantic search, claim extraction, and verification.

## Current Status

| Component | Status |
|---|---|
| PDF ingestion | Implemented |
| MongoDB storage | Implemented |
| Embedding generation | Implemented |
| Semantic search | Implemented |
| Groq claim extraction | Implemented and tested |
| Groq verification | Implemented; live testing pending |
| Contradiction detection | Planned |
| Research gap analysis | Planned |




