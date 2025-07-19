from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
from utils.config import MODEL_DIR, MODEL_FILE, VECTORIZER_FILE

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=300, multi_class='multinomial')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds)
    print(report)

def save_model(model, vectorizer):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("💾 Saving model and vectorizer...")
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print("✅ Saved to:", MODEL_DIR)