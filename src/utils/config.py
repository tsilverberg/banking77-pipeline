import os
from dotenv import load_dotenv

# Dynamically resolve .env path
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

print("🔍 Attempting to load .env from:", env_path)
loaded = load_dotenv(dotenv_path=env_path)
print("✅ .env loaded:", loaded)

# Validate and resolve
def get_env_or_throw(key):
    val = os.getenv(key)
    if not val:
        raise ValueError(f"🚨 .env missing required key: {key}")
    return val

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

DATA_BASE_PATH = data_path
TRAIN_JSON = data_path(get_env_or_throw("TRAIN_JSON"))
TEST_JSON = data_path(get_env_or_throw("TEST_JSON"))
DATA_SOURCE_REPO_URL = get_env_or_throw("DATA_SOURCE_REPO_URL")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "intent_classifier.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
