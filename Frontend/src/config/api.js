/**
 * Proposed FastAPI contract. Update this single file once backend routes are
 * confirmed; no service URL, token, or secret belongs in UI components.
 */
export const API_CONFIG = {
  baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
  endpoints: {
    papers: '/api/papers',
    paperUpload: '/api/papers/upload',
    analysis: '/api/analysis',
  },
  timeoutMs: 30_000,
};

export const apiUrl = (path) => `${API_CONFIG.baseUrl.replace(/\/$/, '')}${path}`;
