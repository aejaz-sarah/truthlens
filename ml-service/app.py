import os
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).with_name("model.joblib")))
model = None
model_error = None

try:
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)
    else:
        model_error = "Trained model not found. Add the LIAR TSV files and run train_model.py."
except Exception as exc:
    model_error = f"Could not load model: {exc}"


@app.get("/health")
def health():
    return jsonify({"available": True, "model_loaded": model is not None, "message": model_error})


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": model_error or "Trained model is unavailable."}), 503
    payload = request.get_json(silent=True) or {}
    statement = payload.get("statement")
    if not isinstance(statement, str) or not statement.strip():
        return jsonify({"error": "Enter a non-empty statement to analyze."}), 400
    try:
        clean_statement = " ".join(statement.split())
        prediction = int(model.predict([clean_statement])[0])
        probabilities = model.predict_proba([clean_statement])[0]
        fake_index = list(model.classes_).index(1)
        fake_probability = float(probabilities[fake_index])
        return jsonify({
            "label": "Fake" if prediction == 1 else "Real",
            "fakeProbability": fake_probability,
            "statement": statement.strip(),
        })
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "The model could not analyze this statement."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5001")), debug=False)
