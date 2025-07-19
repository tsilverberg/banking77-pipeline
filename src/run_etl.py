"""
📦 Banking77 Dataset Import Script

This script downloads the Banking77 dataset from the PolyAI GitHub repo,
converts the CSV files into JSON Lines format, and saves them locally.
All paths and URLs are loaded from a `.env` file for configurability.

Expected .env variables:
- DATA_SOURCE_REPO_URL   (e.g. "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/")
- TRAIN_JSON             (e.g. "data/banking77_train.jsonl")
- TEST_JSON              (e.g. "data/banking77_test.jsonl")
"""

import pandas as pd
import os
from dotenv import load_dotenv
from utils.config import TRAIN_JSON, TEST_JSON, DATA_SOURCE_REPO_URL, DATA_DIR

# Load env variables from .env
load_dotenv()

# GitHub URLs
train_url = DATA_SOURCE_REPO_URL + "train.csv"
test_url  = DATA_SOURCE_REPO_URL + "test.csv"

# 🧪 Log paths for debugging
print("🔍 DATA_SOURCE_REPO_URL:", DATA_SOURCE_REPO_URL)
print("📁 DATA_DIR (base path):", DATA_DIR)
print("📄 TRAIN_JSON:", TRAIN_JSON)
print("📄 TEST_JSON:", TEST_JSON)
print("🌐 Fetching CSVs from:")
print("    🔗", train_url)
print("    🔗", test_url)

# Ensure data/ directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load from remote CSVs
train_df = pd.read_csv(train_url)
test_df  = pd.read_csv(test_url)

# Save as JSONL
train_df.to_json(TRAIN_JSON, orient="records", lines=True)
test_df.to_json(TEST_JSON, orient="records", lines=True)

print("✅ Banking77 train/test saved as JSON Lines:")
print("   -", TRAIN_JSON)
print("   -", TEST_JSON)
