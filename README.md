# TruthLens - Fake News Detection System

A small React, Express, and Flask app based on the supplied LIAR case study. The LIAR partitions are in `ml-service/data/`, and the fitted TF-IDF + Logistic Regression pipeline is saved as `ml-service/model.joblib`, the path loaded by Flask. The app uses live model inference and never substitutes sample outputs.

## Case study methodology

- Dataset: LIAR `train.tsv`, `valid.tsv`, and `test.tsv`; the statement is the only model input.
- Dataset source: [tfs4/liar_dataset](https://github.com/tfs4/liar_dataset).
- Clean by dropping missing labels/statements, trimming and collapsing whitespace, removing empty statements, then de-duplicating case-insensitive statements in train, validation, test order.
- Map `pants-fire`, `false`, and `barely-true` to Fake (1); map `half-true`, `mostly-true`, and `true` to Real (0).
- Fit a scikit-learn pipeline with word TF-IDF unigrams/bigrams, Unicode accent stripping, `min_df=2`, `max_df=0.95`, sublinear term frequency, and L2-normalized vectors, followed by L2 Logistic Regression (`liblinear`, `max_iter=2000`, `random_state=42`).
- Select C from 0.1, 1, and 10 by validation macro F1; the report selected C=1. The held-out test metrics reported by the case study are 61.60% accuracy, 0.5893 macro F1, and 41.38% Fake recall. These are reported results, not a re-evaluation performed by this application.

## Run on Windows

Install Node.js 18+ and Python 3.10+. Open three terminals in `truthlens`.

### 1. ML service

```powershell
cd ml-service
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

It listens on `http://localhost:5001`. `/health` reports whether `model.joblib` loaded successfully.

To retrain, put the original `train.tsv`, `valid.tsv`, and `test.tsv` in `ml-service/data/`, then run `python train_model.py`. This replaces `model.joblib` and prints the selected C and held-out test metrics. Training does not happen on user requests. The current model selected C=1 and reproduced the report's results: 61.60% test accuracy, 0.5893 macro F1, and 41.38% Fake recall.

### 2. Express API

```powershell
cd backend
npm install
Copy-Item .env.example .env
npm start
```

It listens on `http://localhost:5000`; `GET /api/health` describes ML service availability.

### 3. React frontend

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`).

## Deploy all three services on Vercel

The root `vercel.json` keeps the frontend, Express API, and Flask model service in one Vercel Services deployment. Set the Vercel project's **Root Directory** to this `truthlens` folder and its **Framework Preset** to **Services**. Vercel Services must be enabled for the account/project.

- The `frontend` service is public at `/` and receives all paths except `/api/*`.
- The `backend` service is public at `/api/*` and keeps the existing Express API paths.
- The `ml-service` service is internal only. The backend calls it through the `ML_SERVICE_URL` binding injected by Vercel.

The browser calls `/api` on the same deployment domain, so it does not need a frontend-to-backend binding or a production API hostname. The trained `model.joblib` is alongside `app.py` in the Flask service root. Do not add `ML_SERVICE_URL` manually to Vercel's environment settings; the binding supplies it at function runtime.

The Flask service pins Python 3.12 in `ml-service/.python-version` and pins NumPy, SciPy, scikit-learn, and joblib to match the model's training environment. Keep those versions aligned with the saved model when retraining or changing deployment dependencies.

## API

- `GET /api/health` returns Express status and whether Flask has a model loaded.
- `POST /api/predict` accepts `{ "statement": "..." }`. It forwards the statement to Flask and returns `{ "label": "Fake"|"Real", "fakeProbability": number, "statement": string }` from the fitted pipeline's `predict()` and `predict_proba()`.
- Invalid statements return 400. An unavailable Flask service or missing model returns 503.

## Interpretation

The score is the model's raw, uncalibrated P(Fake) estimate. It is not a verified probability that a claim is false. This model learns patterns in short English political claims from LIAR; it does not retrieve evidence or verify current facts. The study reports Fake recall of only 41.38%, meaning it missed many Fake-labeled test claims.

> This result is an ML-based prediction, not a verified fact-check. Please consult reliable sources before drawing conclusions.
