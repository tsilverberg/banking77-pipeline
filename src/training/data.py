import pandas as pd
from utils.config import TRAIN_JSON, TEST_JSON

def load_banking77_from_json():
    train_df = pd.read_json(TRAIN_JSON, lines=True)
    test_df = pd.read_json(TEST_JSON, lines=True)
    return train_df, test_df
