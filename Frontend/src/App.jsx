import { useRef, useState } from 'react';
import { API_CONFIG } from './config/api';
import { listPapers, submitAnalysis, uploadPapers } from './services/apiClient';
import { exampleAnalysis, examplePapers } from './mocks/exampleResponses';

const SECTIONS = [
  ['methodologies', 'Research methodologies'], ['findings', 'Key findings'], ['limitations', 'Limitations'],
  ['contradictions', 'Contradictions & disagreements'], ['gaps', 'Potential research gaps'],
];
const isPdf = (file) => file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
const displayStatus = (status) => ['ready', 'complete', 'completed'].includes(String(status).toLowerCase()) ? 'Ready' : status || 'Processing';

function Citation({ citation, onOpen }) {
  const source = citation.source || citation.paper_title || citation.filename || 'Source paper';
  return <button className="citation" type="button" onClick={() => onOpen(citation)} aria-label={`Review citation from ${source}${citation.page ? `, page ${citation.page}` : ''}`}>{source}{citation.page ? ` · p. ${citation.page}` : ''}</button>;
}

function Alert({ message, onDismiss }) {
  return message ? <div className={`alert ${message.kind || 'error'}`} role="alert"><span>{message.text}</span><button type="button" onClick={onDismiss} aria-label="Dismiss message">×</button></div> : null;
}

export default function App() {
  const fileInput = useRef(null);
  const [papers, setPapers] = useState([]);
  const [question, setQuestion] = useState('');
  const [analysis, setAnalysis] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [loadingPapers, setLoadingPapers] = useState(false);
  const [analysing, setAnalysing] = useState(false);
  const [alert, setAlert] = useState(null);
  const [citation, setCitation] = useState(null);
  const [isExample, setIsExample] = useState(false);

  const refreshLibrary = async () => {
    setLoadingPapers(true); setAlert(null);
    try { setPapers(await listPapers()); setIsExample(false); } catch (error) { setAlert({ kind: 'error', text: error.message }); } finally { setLoadingPapers(false); }
  };

  const chooseFiles = async (event) => {
    const allFiles = [...event.target.files]; event.target.value = '';
    const valid = allFiles.filter(isPdf); const invalid = allFiles.filter((file) => !isPdf(file));
    if (invalid.length) setAlert({ kind: 'error', text: `${invalid.map((file) => file.name).join(', ')} is not a PDF and was not uploaded.` });
    if (!valid.length) return;
    setUploading(true); setProgress(0); setAlert(null);
    try {
      const uploaded = await uploadPapers(valid, setProgress);
      setPapers((current) => [...uploaded, ...current.filter((old) => !uploaded.some((next) => next.id && next.id === old.id))]);
      setIsExample(false);
      setAlert({ kind: 'success', text: `${valid.length} paper${valid.length === 1 ? ' was' : 's were'} accepted. Processing status will update when the backend reports it.` });
    } catch (error) { setAlert({ kind: 'error', text: error.message }); } finally { setUploading(false); }
  };

  const analyse = async (event) => {
    event.preventDefault();
    if (!question.trim()) return setAlert({ kind: 'error', text: 'Enter a research question before starting an analysis.' });
    if (!papers.length) return setAlert({ kind: 'error', text: 'Upload or load at least one paper before starting an analysis.' });
    setAnalysing(true); setAnalysis(null); setAlert(null);
    try { setAnalysis(await submitAnalysis(question.trim(), papers.map((paper) => paper.id).filter(Boolean))); setIsExample(false); }
    catch (error) { setAlert({ kind: 'error', text: error.message }); } finally { setAnalysing(false); }
  };

  const loadExample = () => {
    setPapers(examplePapers); setAnalysis(exampleAnalysis); setQuestion('What gaps remain in adaptive, low-data scientific models?'); setIsExample(true);
    setAlert({ kind: 'success', text: 'Example data loaded. It is not connected to your backend.' });
  };

  const sections = analysis?.sections || analysis || {};
  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to main content</a>
    <header className="topbar"><div className="brand"><span className="brand-mark" aria-hidden="true">R+</span><div><strong>Research+</strong><small>Explore opportunities</small></div></div><span className="api-indicator">API: {API_CONFIG.baseUrl}</span></header>
    <main id="main" className="workspace">
      <section className="hero"><h1>Find the questions the literature has not answered.</h1><p>Upload a focused paper set, then trace research gaps back to the claims and passages that support them.</p></section>
      <Alert message={alert} onDismiss={() => setAlert(null)} />
      <section className="panel upload-panel" aria-labelledby="papers-heading">
        <div className="section-heading"><div><p className="eyebrow">01 · Paper library</p><h2 id="papers-heading">Build your evidence set</h2><p>PDF papers only. Files are sent directly to your configured FastAPI backend.</p></div><div className="actions"><button className="secondary-button" type="button" onClick={loadExample}>Load example data</button><button className="secondary-button" type="button" onClick={refreshLibrary} disabled={loadingPapers}>{loadingPapers ? 'Refreshing…' : 'Refresh library'}</button></div></div>
        <input ref={fileInput} className="visually-hidden" type="file" accept="application/pdf,.pdf" multiple onChange={chooseFiles} />
        <button className="dropzone" type="button" onClick={() => fileInput.current?.click()} disabled={uploading}><span className="upload-icon" aria-hidden="true">↑</span><strong>{uploading ? `Uploading ${progress}%` : 'Choose PDF papers'}</strong><span>{uploading ? 'Keep this page open while files upload.' : 'Select one or more files from your computer.'}</span>{uploading && <span className="progress" aria-label={`Upload progress ${progress}%`}><i style={{ width: `${progress}%` }} /></span>}</button>
        {papers.length ? <div className="library-list" aria-label="Uploaded papers">{papers.map((paper, index) => <article className="paper-row" key={paper.id || paper.filename || index}><div className="file-glyph" aria-hidden="true">PDF</div><div className="paper-details"><strong>{paper.title || paper.filename || 'Untitled paper'}</strong><span>{[paper.authors?.join(', '), paper.year, paper.pages ? `${paper.pages} pages` : null].filter(Boolean).join(' · ') || 'Metadata will appear here when returned by the backend.'}</span></div><span className={`status ${String(paper.status || '').toLowerCase()}`}>{displayStatus(paper.status)}</span></article>)}</div> : <div className="empty-state"><strong>Your library is empty.</strong><span>Upload PDFs, refresh a backend library, or load the clearly labelled example data.</span></div>}
      </section>
      <section className="panel query-panel" aria-labelledby="analysis-heading"><div><p className="eyebrow">02 · Gap analysis</p><h2 id="analysis-heading">Ask a precise research question</h2></div><form onSubmit={analyse}><label htmlFor="question">Research question or topic</label><textarea id="question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="For example: Where does the evidence on a method disagree, and what would resolve it?" rows="4" /><div className="query-footer"><span>{papers.length} paper{papers.length === 1 ? '' : 's'} in scope</span><button className="primary-button" type="submit" disabled={analysing || !papers.length}>{analysing ? 'Analyzing evidence…' : 'Analyze literature'}</button></div></form></section>
      <section className="results" aria-labelledby="results-heading"><div className="results-heading"><div><p className="eyebrow">03 · Evidence report</p><h2 id="results-heading">Analysis</h2></div>{isExample && <span className="example-badge">Example data</span>}</div>
        {analysing ? <div className="loading-state" role="status"><span className="spinner" aria-hidden="true" />Retrieving and organizing evidence…</div> : !analysis ? <div className="empty-state large"><strong>No analysis yet.</strong><span>Ask a question after adding papers to generate an evidence-grounded report.</span></div> : <>
          {analysis.evidence_sufficient === false && <div className="notice" role="status"><strong>Evidence may be insufficient.</strong> {analysis.insufficiency_reason || 'The retrieved passages do not support a confident conclusion. Consider adding more relevant papers or narrowing the question.'}</div>}
          {analysis.answer && <div className="summary"><p className="eyebrow">Synthesis</p><p>{analysis.answer}</p></div>}
          <div className="analysis-grid">{SECTIONS.map(([key, title]) => <article className="analysis-card" key={key}><h3>{title}</h3>{sections[key]?.length ? <ol>{sections[key].map((item, index) => <li key={item.id || index}><p>{item.claim || item.text || String(item)}</p>{item.citations?.length ? <div className="citations">{item.citations.map((source, sourceIndex) => <Citation key={source.id || sourceIndex} citation={source} onOpen={setCitation} />)}</div> : <span className="no-citation">No source location supplied</span>}</li>)}</ol> : <p className="muted">No evidence was returned for this section.</p>}</article>)}</div>
        </>}
      </section>
    </main>
    {citation && <div className="modal-backdrop" role="presentation" onMouseDown={() => setCitation(null)}><section className="citation-modal" role="dialog" aria-modal="true" aria-labelledby="citation-title" onMouseDown={(event) => event.stopPropagation()}><button className="close-button" type="button" onClick={() => setCitation(null)} aria-label="Close cited passage">×</button><p className="eyebrow">Cited evidence</p><h2 id="citation-title">{citation.source || citation.paper_title || citation.filename || 'Source paper'}</h2>{citation.page && <p className="page-label">Page {citation.page}</p>}{citation.excerpt || citation.text ? <blockquote>{citation.excerpt || citation.text}</blockquote> : <p className="muted">The backend supplied this citation without a passage excerpt.</p>}</section></div>}
  </div>;
}
