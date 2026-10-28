import { useEffect, useState } from 'react';

const examples = [
  'The city council approved the annual budget after a public vote.',
  'The unemployment rate fell by two percentage points last year.',
  'The government has abolished every tax in the country overnight.'
];

export default function App() {
  const [statement, setStatement] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(null);

  useEffect(() => {
    fetch('/api/health').then((response) => response.json())
      .then((health) => setModelReady(Boolean(health.mlService?.model_loaded)))
      .catch(() => setModelReady(false));
  }, []);

  async function analyze(event) {
    event.preventDefault();
    if (!statement.trim() || loading) return;
    setLoading(true); setError(''); setResult(null);
    try {
      const response = await fetch('/api/predict', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ statement })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Could not analyze this statement.');
      setResult(data);
    } catch (requestError) {
      setError(requestError.message || 'Could not reach the API. Check that the Express and Flask services are running.');
    } finally { setLoading(false); }
  }

  function clear() { setStatement(''); setResult(null); setError(''); }

  return <div className="page-shell">
    <header className="topbar">
      <a className="brand" href="#top" aria-label="TruthLens home"><span className="brand-mark">T</span><span>TruthLens</span></a>
      <nav aria-label="Main navigation"><a href="#analyzer">Analyzer</a><a href="#about">About the model</a></nav>
      <span className={`service-pill ${modelReady ? 'ready' : ''}`}><i />{modelReady === null ? 'Checking model' : modelReady ? 'Model connected' : 'Model unavailable'}</span>
    </header>

    <main id="top">
      <section className="intro">
        <p className="eyebrow"><span /> A classroom project in text classification</p>
        <h1>Look closer at<br /><em>what a claim says.</em></h1>
        <p className="lede">TruthLens uses a text classification model trained on LIAR claims to explore how language patterns relate to fact-checker labels.</p>
      </section>

      <section className="workspace" id="analyzer">
        <form className="input-card" onSubmit={analyze}>
          <div className="card-heading"><div><span className="step">01</span><h2>Enter a statement</h2></div><span className="input-hint">English claims · up to 10,000 characters</span></div>
          <textarea value={statement} onChange={(event) => setStatement(event.target.value)} maxLength={10000} placeholder="Paste a short news claim or political statement here…" aria-label="News statement" />
          <div className="examples"><span>Try an example</span>{examples.map((example, index) => <button key={example} type="button" onClick={() => { setStatement(example); setError(''); setResult(null); }}>0{index + 1}</button>)}</div>
          {error && <div className="error-message" role="alert">{error}</div>}
          <div className="form-actions"><span className="char-count">{statement.length.toLocaleString()} / 10,000</span><div><button className="clear-button" type="button" onClick={clear} disabled={!statement && !result}>Clear</button><button className="analyze-button" type="submit" disabled={!statement.trim() || loading}>{loading ? <><span className="spinner" />Analyzing</> : <>Analyze statement <span aria-hidden="true">→</span></>}</button></div></div>
        </form>

        <aside className="result-card" aria-live="polite">
          <div className="card-heading"><div><span className="step">02</span><h2>Model output</h2></div></div>
          {result ? <div className="result-content">
            <div className={`label-badge ${result.label.toLowerCase()}`}><span />{result.label} pattern</div>
            <h3>{result.label}</h3>
            <p className="probability-label">Model estimate for the Fake class</p>
            <div className="probability"><strong>{(result.fakeProbability * 100).toFixed(1)}<small>%</small></strong><div className="meter"><span style={{ width: `${Math.min(100, Math.max(0, result.fakeProbability * 100))}%` }} /></div><div className="meter-ends"><span>0</span><span>100</span></div></div>
            <div className="submitted"><span>SUBMITTED STATEMENT</span><p>“{result.statement}”</p></div>
            <p className="result-disclaimer">This score is uncalibrated and reflects patterns in the training data, not a verified chance that the claim is false.</p>
          </div> : <div className="empty-result"><div className="empty-icon">↗</div><h3>Your result will appear here</h3><p>Submit a statement to see the model’s label and its raw Fake class score.</p><span className="empty-divider" /><p className="availability">{modelReady ? 'The model service is connected.' : 'A trained model is not currently available. Inference will become available after the LIAR model is trained.'}</p></div>}
        </aside>
      </section>

      <section className="performance" aria-label="Reported model performance">
        <div className="performance-title"><span>CASE STUDY RESULTS</span><p>Held-out LIAR test set</p></div>
        <div className="metric"><strong>61.60<span>%</span></strong><span>Test accuracy</span></div>
        <div className="metric"><strong>0.5893</strong><span>Macro F1-score</span></div>
        <div className="metric caution"><strong>41.38<span>%</span></strong><span>Fake recall</span></div>
        <p className="metric-footnote">Reported by the case study · 1,263 test claims</p>
      </section>

      <section className="about" id="about">
        <div className="about-heading"><p className="eyebrow">HOW IT WORKS</p><h2>A simple text<br />classification pipeline.</h2></div>
        <div className="about-copy"><p>The LIAR dataset contains short political claims labeled by professional fact-checkers. TruthLens maps <b>pants-fire, false, and barely-true</b> to Fake; <b>half-true, mostly-true, and true</b> to Real.</p><p>Text becomes TF-IDF word unigrams and bigrams. An L2-regularized Logistic Regression model (C = 1) learns patterns from those labels. The case study selected C using validation macro F1.</p><p className="caution-note"><span>!</span><span>Fake recall was 41.38% in the reported test results. The model misses many Fake-labeled claims and does not check evidence, context, or current facts.</span></p></div>
      </section>
      <footer><a className="brand" href="#top"><span className="brand-mark">T</span><span>TruthLens</span></a><p>This result is an ML-based prediction, not a verified fact-check. Please consult reliable sources before drawing conclusions.</p><span>EDUCATIONAL DEMO</span></footer>
    </main>
  </div>;
}
