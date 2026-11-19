import { API_CONFIG, apiUrl } from '../config/api';

export class ApiError extends Error {
  constructor(message, status, details) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), API_CONFIG.timeoutMs);
  try {
    const response = await fetch(apiUrl(path), { ...options, signal: controller.signal });
    const type = response.headers.get('content-type') || '';
    const payload = type.includes('application/json') ? await response.json() : await response.text();
    if (!response.ok) throw new ApiError(payload?.detail || payload?.message || `Request failed (${response.status}).`, response.status, payload);
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
  return Array.isArray(payload) ? payload : payload.papers || [];
}

/** Proposed: POST /api/papers multipart/form-data (`files`) -> { papers: Paper[] }. */
export function uploadPapers(files, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    const xhr = new XMLHttpRequest();
    xhr.open('POST', apiUrl(API_CONFIG.endpoints.papers));
    xhr.timeout = API_CONFIG.timeoutMs;
    xhr.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(Math.round((event.loaded / event.total) * 100)); };
    xhr.onerror = () => reject(new ApiError('Could not reach the API. Confirm the backend URL and its CORS settings.'));
    xhr.ontimeout = () => reject(new ApiError('The upload timed out. Please try again.'));
    xhr.onload = () => {
      let payload = {};
      try { payload = xhr.responseText ? JSON.parse(xhr.responseText) : {}; } catch { /* backend returned non-JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(Array.isArray(payload) ? payload : payload.papers || []);
      else reject(new ApiError(payload.detail || payload.message || `Upload failed (${xhr.status}).`, xhr.status, payload));
    };
    xhr.send(form);
  });
}

/** Proposed: POST /api/analysis { question, paper_ids? } -> AnalysisResponse */
export const submitAnalysis = (question, paperIds) => request(API_CONFIG.endpoints.analysis, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ question, paper_ids: paperIds.length ? paperIds : undefined }),
});
