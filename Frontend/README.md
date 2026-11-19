# LitScope frontend

Local React frontend for an **AI-Powered Scientific Literature Gap Discovery System**. It has no database implementation, PDF processing, vector retrieval, LLM integration, API keys, or backend secrets. The browser calls only the configured FastAPI HTTP service.

## Folder tree

```text
Frontend/
├── .env.example
├── index.html
├── package.json
├── vite.config.js
└── src/
    ├── App.jsx
    ├── main.jsx
    ├── styles.css
    ├── config/api.js
    ├── services/apiClient.js
    └── mocks/exampleResponses.js
```

## Windows 11 / VS Code

Install the current Node.js LTS release. In VS Code, open this `Frontend` folder and run:

```powershell
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`, in Chrome or Brave. Build the production version with:

```powershell
npm run build
npm run preview
```

Optional: copy `.env.example` to `.env`, set `VITE_API_BASE_URL` to your FastAPI service, then restart the dev server.

## Confirmed paper API contract

All paths are centralised in `src/config/api.js`. The analysis route remains configurable for a later backend phase.

| Operation | Request | Successful response |
| --- | --- | --- |
| List papers | `GET /api/papers` | `{ "papers": [Paper], "total": 0, "skip": 0, "limit": 20 }` |
| Upload one PDF | `POST /api/papers/upload`, multipart field `file` | `PaperUploadResponse` |
| Analyze | `POST /api/analysis`, `{ "question": "...", "paper_ids": ["..."] }` | `AnalysisResponse` when implemented |

When multiple PDFs are selected, the frontend sends one request per PDF to the single-file upload endpoint and keeps completed uploads visible if a later file fails.

`Paper` should include `id`, `filename` or `title`, `status` (`queued`, `processing`, `ready`, or `failed`), and optional `authors`, `year`, `pages`. The frontend accepts FastAPI `{ "detail": "..." }` errors or `{ "message": "..." }` errors.

```json
{
  "answer": "Brief evidence-grounded synthesis.",
  "evidence_sufficient": true,
  "insufficiency_reason": "Optional explanation when false.",
  "sections": {
    "methodologies": [{
      "claim": "A claim grounded in retrieved evidence.",
      "citations": [{
        "paper_id": "paper-123",
        "source": "Paper title or filename",
        "page": 7,
        "excerpt": "Optional cited passage."
      }]
    }],
    "findings": [], "limitations": [], "contradictions": [], "gaps": []
  }
}
```

The **Load example data** button is the only mock-data path and labels its results as examples; it never means that a backend is connected.

## Backend details needed for final integration

- Actual FastAPI base URL, endpoint paths, and `/api` prefix decision.
- Exact uploads, JSON schema, paper-ID field, status values, upload limits, and processing-status update method (polling, SSE, or WebSocket).
- Final analysis/citation fields, page-number convention, and optional safe PDF/passage URL.
- Development and deployment CORS origins, plus authentication design if required.
