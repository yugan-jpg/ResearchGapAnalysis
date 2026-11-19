import { afterEach, describe, expect, it, vi } from 'vitest';
import { listPapers, uploadPapers } from './apiClient';

const backendPaper = {
  paper_id: 'paper-123',
  filename: 'study.pdf',
  total_pages: 12,
  pages_with_text: 12,
  total_chunks: 24,
  status: 'processed',
};

class MockXmlHttpRequest {
  static responses = [];
  static requests = [];

  constructor() {
    this.upload = {};
    this.timeout = 0;
  }

  open(method, url) {
    this.method = method;
    this.url = url;
  }

  send(form) {
    this.form = form;
    MockXmlHttpRequest.requests.push(this);
    this.upload.onprogress?.({ lengthComputable: true, loaded: 100, total: 100 });
    const response = MockXmlHttpRequest.responses.shift();
    if (response.type === 'network-error') {
      this.onerror?.();
      return;
    }
    this.status = response.status;
    this.responseText = JSON.stringify(response.body);
    this.onload?.();
  }
}

function pdf(name, size = 100) {
  const file = new File(['pdf'], name, { type: 'application/pdf' });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

describe('paper API client', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    MockXmlHttpRequest.responses = [];
    MockXmlHttpRequest.requests = [];
    globalThis.XMLHttpRequest = undefined;
  });

  it('lists and normalizes persisted papers for the library', async () => {
    globalThis.window = globalThis;
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      headers: new Headers({ 'content-type': 'application/json' }),
      json: async () => ({ papers: [{ ...backendPaper, original_filename: 'study.pdf', metadata: { title: 'Study title' } }] }),
    });

    const papers = await listPapers();

    expect(globalThis.fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/api/papers', expect.any(Object));
    expect(papers).toEqual([expect.objectContaining({ id: 'paper-123', filename: 'study.pdf', title: 'Study title', pages: 12 })]);
  });

  it('uploads one PDF to the confirmed endpoint and field name', async () => {
    globalThis.XMLHttpRequest = MockXmlHttpRequest;
    MockXmlHttpRequest.responses = [{ status: 201, body: backendPaper }];

    const result = await uploadPapers([pdf('study.pdf')]);

    expect(result.failures).toEqual([]);
    expect(result.papers).toEqual([expect.objectContaining({ id: 'paper-123', filename: 'study.pdf', pages: 12 })]);
    expect(MockXmlHttpRequest.requests[0].method).toBe('POST');
    expect(MockXmlHttpRequest.requests[0].url).toBe('http://127.0.0.1:8000/api/papers/upload');
    expect(MockXmlHttpRequest.requests[0].form.get('file').name).toBe('study.pdf');
    expect(MockXmlHttpRequest.requests[0].form.get('files')).toBeNull();
  });

  it('uploads multiple selected PDFs individually and reports aggregate progress', async () => {
    globalThis.XMLHttpRequest = MockXmlHttpRequest;
    MockXmlHttpRequest.responses = [
      { status: 201, body: backendPaper },
      { status: 201, body: { ...backendPaper, paper_id: 'paper-456', filename: 'second.pdf' } },
    ];
    const progress = vi.fn();

    const result = await uploadPapers([pdf('study.pdf'), pdf('second.pdf')], progress);

    expect(MockXmlHttpRequest.requests).toHaveLength(2);
    expect(MockXmlHttpRequest.requests.every((request) => request.url.endsWith('/api/papers/upload'))).toBe(true);
    expect(result.papers.map((paper) => paper.id)).toEqual(['paper-123', 'paper-456']);
    expect(progress).toHaveBeenLastCalledWith(100);
  });

  it('returns a meaningful backend validation error without discarding other results', async () => {
    globalThis.XMLHttpRequest = MockXmlHttpRequest;
    MockXmlHttpRequest.responses = [
      { status: 201, body: backendPaper },
      { status: 422, body: { detail: 'No readable text was found.' } },
    ];

    const result = await uploadPapers([pdf('study.pdf'), pdf('scan.pdf')]);

    expect(result.papers).toHaveLength(1);
    expect(result.failures).toEqual([expect.objectContaining({ filename: 'scan.pdf', error: expect.objectContaining({ message: 'No readable text was found.', status: 422 }) })]);
  });
});
