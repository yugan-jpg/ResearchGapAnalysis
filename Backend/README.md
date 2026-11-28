# Scientific Literature Gap Discovery — Backend

FastAPI backend for scientific PDF processing, semantic search, claim extraction, verification and cross-paper comparison using RAG.

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
- Cross-paper structural similarity using claim cosine similarity and shared scientific terms
- Trusted-claim selection and claim-aware retrieval of comparison candidates
- Independent relationship assessments and potential scientific mismatch signals

## Setup

From the `Backend` directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
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
| POST | `/api/comparisons/similar` |
| POST | `/api/comparisons/detect` |
| GET | `/api/comparisons` |

Maintenance scripts for historical repair, duplicate resolution and targeted re-verification are available in `scripts/`. Use `--help` for usage.

## Cross-paper comparisons

Run `python scripts/initialize_database.py` once to add the ordinary indexes. In Atlas, create or update the `paper_chunks_vector_index` Vector Search index on `paper_chunks` using [atlas_vector_index.json](atlas_vector_index.json), then wait until it is ready. The definition uses a 384-dimensional cosine vector field named `embedding` and filter fields `paper_id`, `chunk_id`, `embedding_model`, `embedding_dimensions` and `embedding_status`. Existing indexes need the `chunk_id` filter for larger target papers. If using a different index name, configure `VECTOR_SEARCH_INDEX_NAME` accordingly.

In Swagger, choose two independent publications from `GET /api/papers`. Each needs accepted, independently verified claims with traceable evidence. Larger target papers need indexed chunk embeddings from the configured model. If necessary, run `POST /api/embeddings/backfill` with `{"paper_id":"target-id","limit":50,"force":false}` until `remaining_chunks` is zero.

The comparison selector requires accepted verification, evidence support, numerical consistency, completeness, category correctness, verifier provenance and an exact quotation in the recorded paper, chunk and page. It excludes claims requiring re-verification and unresolved or confirmed redundant duplicates. Publication identity is checked using recorded version links, DOI and file hashes.

1. **Discover candidates** using `POST /api/comparisons/similar`. Substitute your paper IDs.

   ```json
   {"source_paper_id":"source-id","target_paper_id":"target-id","limit":2,"min_similarity":0.60}
   ```

   The response contains `scanned` source records, `queued` new or refreshed pairs, `comparisons` and `next_cursor`. On the next request, pass that cursor as `after_claim_id`. Stop at a null cursor. An empty page can still have a next cursor. Rediscover from the first page after adding or re-verifying claims. Reversing the paper order can surface different Atlas neighbors. An unordered claim pair shares one database record, preventing duplicate queue entries.

2. **Evaluate a small batch** using `POST /api/comparisons/detect`.

   ```json
   {"paper_id":"source-id","other_paper_id":"target-id","limit":1,"retry_failed":false}
   ```

   A successful new pair usually costs two Groq completions before retries. Qwen 3.8 27B and GPT-OSS-120B independently assess the same claims and passages. The response counts attempted, completed, failed and stale pairs. For example, a completed item may contain `"relationship":"contextual_differences"`, `"mismatch_types":["population_context"]` and `"verification_status":"verified"`. Each side includes the original claim, passage, paper, page and chunk. Both model judgments include exact quotations. Disagreement or insufficient support leads to review.

   Repeat to drain the queue. Once eligible work is exhausted the response is `{"attempted":0,"completed":0,"failed":0,"stale":0,"comparisons":[]}`. On a provider failure the batch stops. Inspect `failure_reason`, then submit `retry_failed:true` when ready. A saved primary judgment is reused if the verifier failed. A crashed worker's reservation expires automatically.

3. **Retrieve results** using `GET /api/comparisons`. For example, supply `paper_id=source-id`, `relationship=contextual_differences`, `mismatch_type=population_context`, `skip=0` and `limit=20`. The response contains `total`, `skip`, `limit` and `comparisons`. Filters apply to stored classifications. A result whose source quotation or eligibility changed retains its historical judgments and returns `sources_current:false` and `verification_status:stale`.

The `contradictions` collection holds processing state and six relationship types: `agreement`, `direct_contradiction`, `partial_contradiction`, `complementary_findings`, `contextual_differences` and `insufficient_evidence`. A future consumer should require `sources_current:true` and `verification_status:verified`. The `claim_scope`, `evidence_method`, `population_context` and `cross_field` mismatch labels are potential signals for manual investigation.

Optional settings in `.env.example` control discovery and processing:

| Setting | Default | Purpose |
|---|---:|---|
| `COMPARISON_MIN_SIMILARITY` | 0.60 | Minimum claim-to-claim cosine score |
| `COMPARISON_NEIGHBORS` | 5 | Maximum target chunks and returned pairs per source claim |
| `COMPARISON_DISCOVERY_LIMIT` | 5 | Maximum source claims scanned per request |
| `COMPARISON_DETECTION_LIMIT` | 2 | Maximum queued pairs evaluated per request |

When trusted claims occupy no more chunks than the neighbor limit, discovery checks those chunks directly. Larger targets use bounded Atlas retrieval filtered to trusted claim-bearing chunks. Candidates must pass the cosine threshold and a shared-term structural screen. The target scan is limited to 100,000 eligible records and 10,000 trusted chunks per request.

English-language cues, vocabulary differences, missing scientific context and bounded retrieval can cause relevant pairs to be missed or unrelated pairs to be shortlisted. Mismatch classification depends on the supplied passages and model judgments. Agreement between models does not establish scientific correctness. Review the original studies and their methods, populations and measurement conditions before interpreting a relationship or potential gap signal. Recorded publication metadata cannot establish whether separately published studies reused the same cohort. Research gap detection has not been fully validated.

## Testing

```powershell
python -m pytest -q
```

The live Groq smoke test is opt-in via `RUN_GROQ_QUALITY_SMOKE=1`. Comparison tests use mocked Atlas and Groq responses.



