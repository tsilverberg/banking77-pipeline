"""
🚀 Banking77 Intent Classification API

This FastAPI app loads a trained intent classification model (Logistic Regression)
and exposes a `/predict` endpoint to classify user queries into banking-related intents.

Run with:
    uvicorn app:app --reload

Requires:
- Trained model and vectorizer (see config.py)
- FastAPI + Uvicorn
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from utils.config import MODEL_FILE, VECTORIZER_FILE

# Load model + vectorizer
model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)

# Init FastAPI
app = FastAPI(title="Banking77 Intent Classifier")

# Input schema
class Query(BaseModel):
    text: str

@app.get("/")
def health_check():
    return {"status": "up"}

@app.post("/predict")
def predict(query: Query):
    X = vectorizer.transform([query.text])
    intent = model.predict(X)[0]
    return {
        "text": query.text,
        "predicted_intent": intent
    }
