"""
🎯 run_training.py

This script orchestrates the full training pipeline:
1. Load data from JSONL (path from .env)
2. Transform text into features using TF-IDF
3. Train a Logistic Regression classifier
4. Evaluate performance
5. Save model and vectorizer artifacts
"""

import os
from dotenv import load_dotenv
from training.data import load_banking77_from_json
from training.features import extract_features
from training.model import train_model, evaluate_model, save_model

# Load environment variables
load_dotenv()

def main():
    print("📦 Loading Banking77 dataset...")
    train_df, test_df = load_banking77_from_json()

    print("🔧 Extracting TF-IDF features...")
    X_train, X_test, vectorizer = extract_features(train_df["text"], test_df["text"])

    print("🤖 Training Logistic Regression model...")
    model = train_model(X_train, train_df["category"])

    print("📊 Evaluating model performance...")
    evaluate_model(model, X_test, test_df["category"])

    print("💾 Saving model and vectorizer...")
    save_model(model, vectorizer)

    print("✅ Training pipeline completed!")

if __name__ == "__main__":
    main()
