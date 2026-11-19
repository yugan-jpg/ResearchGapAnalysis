import { API_CONFIG, apiUrl } from '../config/api';

export class ApiError extends Error {
  constructor(message, status, details) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

export function messageFromPayload(payload, fallback) {
  if (typeof payload?.detail === 'string') return payload.detail;
  if (typeof payload?.detail?.message === 'string') return payload.detail.message;
  if (typeof payload?.message === 'string') return payload.message;
  return fallback;
}

export function normalisePaper(paper) {
  const metadata = paper.metadata || {};
  return {
    ...paper,
    id: paper.id || paper.paper_id,
    paper_id: paper.paper_id || paper.id,
    filename: paper.filename || paper.original_filename,
    title: paper.title || metadata.title,
    authors: paper.authors || (metadata.author ? [metadata.author] : undefined),
    pages: paper.pages || paper.total_pages,
  };
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), API_CONFIG.timeoutMs);
  try {
    const response = await fetch(apiUrl(path), { ...options, signal: controller.signal });
    const type = response.headers.get('content-type') || '';
    const payload = type.includes('application/json') ? await response.json() : await response.text();
    if (!response.ok) throw new ApiError(messageFromPayload(payload, `Request failed (${response.status}).`), response.status, payload);
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') throw new ApiError('The request timed out. Confirm that the FastAPI service is running and try again.');
    if (error instanceof ApiError) throw error;
    throw new ApiError('Could not reach the API. Confirm the backend URL and its CORS settings.');
  } finally { window.clearTimeout(timer); }
}

/** Proposed: GET /api/papers -> { papers: Paper[] } */
export async function listPapers() {
  const payload = await request(API_CONFIG.endpoints.papers);
  const papers = Array.isArray(payload) ? payload : payload.papers || [];
  return papers.map(normalisePaper);
}

function uploadOnePaper(file, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', apiUrl(API_CONFIG.endpoints.paperUpload));
    xhr.timeout = API_CONFIG.timeoutMs;
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(Math.round((event.loaded / event.total) * 100)); };
    xhr.onerror = () => reject(new ApiError('Could not reach the API. Confirm the backend URL and its CORS settings.'));
    xhr.ontimeout = () => reject(new ApiError('The upload timed out. Please try again.'));
    xhr.onload = () => {
      let payload = {};
      try { payload = xhr.responseText ? JSON.parse(xhr.responseText) : {}; } catch { /* backend returned non-JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(normalisePaper(payload));
      else reject(new ApiError(messageFromPayload(payload, `Upload failed (${xhr.status}).`), xhr.status, payload));
    };
    xhr.send(form);
  });
}

/**
 * The FastAPI API accepts one `file` at a time. Selected files are therefore
 * uploaded individually, retaining progress while allowing successful papers
 * to appear even if another file fails validation.
 */
export async function uploadPapers(files, onProgress, onPaperUploaded) {
  const queue = [...files];
  const totalBytes = queue.reduce((total, file) => total + (file.size || 0), 0);
  let completedBytes = 0;
  const papers = [];
  const failures = [];

  for (const file of queue) {
    try {
      const paper = await uploadOnePaper(file, (fileProgress) => {
        const uploadedBytes = totalBytes ? ((file.size || 0) * fileProgress) / 100 : 0;
        const progress = totalBytes ? Math.round(((completedBytes + uploadedBytes) / totalBytes) * 100) : 0;
        onProgress?.(progress);
      });
      completedBytes += file.size || 0;
      papers.push(paper);
      onPaperUploaded?.(paper);
      onProgress?.(totalBytes ? Math.round((completedBytes / totalBytes) * 100) : 100);
    } catch (error) {
      completedBytes += file.size || 0;
      failures.push({ filename: file.name, error: error instanceof ApiError ? error : new ApiError('The upload failed unexpectedly.') });
      onProgress?.(totalBytes ? Math.round((completedBytes / totalBytes) * 100) : 100);
    }
  }
  return { papers, failures };
}

/** Proposed: POST /api/analysis { question, paper_ids? } -> AnalysisResponse */
export const submitAnalysis = (question, paperIds) => request(API_CONFIG.endpoints.analysis, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, paper_ids: paperIds.length ? paperIds : undefined }),
});
