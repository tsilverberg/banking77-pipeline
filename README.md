# 🏦 Banking77 Intent Classification Pipeline

This project demonstrates a **mini MLOps pipeline** for banking-related intent classification using the [Banking77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets). It covers:

- ✅ Data ingestion & transformation  
- ✅ Feature engineering (TF-IDF)  
- ✅ Model training (Logistic Regression)  
- ✅ Model persistence (joblib)  
- ✅ Mock deployment (FastAPI API)

---

## 📁 Project Structure

```
banking77_pipeline/
├── .env
├── data/
├── models/
├── src/
│   ├── app.py               ← FastAPI app for inference
│   ├── run_etl.py           ← Extracts and converts CSV → JSONL
│   ├── run_training.py      ← Trains model and saves artifacts
│   ├── utils/
│   │   └── config.py        ← Centralized path + env handling
│   └── training/
│       ├── data.py
│       └── model.py
└── README.md
```

---

## 📦 Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd banking77_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```dotenv
# .env file at project root
DATA_SOURCE_REPO_URL=https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/
TRAIN_JSON=banking77_train.jsonl
TEST_JSON=banking77_test.jsonl
MODEL_FILE=models/intent_classifier.pkl
VECTORIZER_FILE=models/vectorizer.pkl
```

---

## 🔄 Run the Pipeline

### ✅ Extract & Save JSONL

```bash
python3 src/run_etl.py
```

### ✅ Train the model

```bash
python3 src/run_training.py
```

---

## 🚀 Run the API

### Start server

```bash
cd src/
uvicorn app:app --reload
```

### Open in browser

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Example request

```json
POST /predict
{
  "text": "How can I reset my PIN?"
}
```

### Example response

```json
{
  "text": "How can I reset my PIN?",
  "predicted_intent": "pin_change"
}
```

---

## 🧪 Next Steps

- Add `confidence_score` via `model.predict_proba`
- Write unit tests for `data.py` and `model.py`
- Add Dockerfile for deployability
- CI integration or model versioning

---

## 📚 Credits

- [PolyAI Banking77 Dataset](https://github.com/PolyAI-LDN/task-specific-datasets)
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)
