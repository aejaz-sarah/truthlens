import 'dotenv/config';
import express from 'express';
import cors from 'cors';

const app = express();
const port = Number(process.env.PORT || 5000);
app.use(cors({ origin: process.env.FRONTEND_ORIGIN || 'http://localhost:5173' }));
app.use(express.json({ limit: '32kb' }));

// Vercel injects ML_SERVICE_URL from the backend -> ml-service binding.
// The localhost fallback keeps the three-service Windows development setup working.
function mlServiceEndpoint(path) {
  const baseUrl = process.env.ML_SERVICE_URL || 'http://127.0.0.1:5001';
  return new URL(path, baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`);
}

app.get('/api/health', async (_req, res) => {
  try {
    const response = await fetch(mlServiceEndpoint('/health'), { signal: AbortSignal.timeout(2500) });
    const health = await response.json();
    res.json({ status: 'ok', mlService: response.ok ? health : { available: false, model_loaded: false } });
  } catch {
    res.json({ status: 'ok', mlService: { available: false, model_loaded: false } });
  }
});

app.post('/api/predict', async (req, res) => {
  const { statement } = req.body ?? {};
  if (typeof statement !== 'string' || !statement.trim()) {
    return res.status(400).json({ error: 'Enter a non-empty statement to analyze.' });
  }
  if (statement.trim().length > 10000) {
    return res.status(400).json({ error: 'Please keep the statement under 10,000 characters.' });
  }
  try {
    const response = await fetch(mlServiceEndpoint('/predict'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ statement: statement.trim() }), signal: AbortSignal.timeout(15000)
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) return res.status(response.status === 503 ? 503 : 502).json({ error: result.error || 'The ML service could not make a prediction.' });
    return res.json(result);
  } catch {
    return res.status(503).json({ error: 'The ML service is unavailable. Start Flask and make sure its model is loaded.' });
  }
});

app.use((err, _req, res, _next) => {
  if (err instanceof SyntaxError) return res.status(400).json({ error: 'Request body must be valid JSON.' });
  console.error(err);
  return res.status(500).json({ error: 'Unexpected server error.' });
});

app.listen(port, () => console.log(`TruthLens API listening at http://localhost:${port}`));
