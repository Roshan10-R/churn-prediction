import os
import joblib

from src.config import MODEL_PATH


def test_debug_model():
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"Exists: {os.path.exists(MODEL_PATH)}")

    model = joblib.load(MODEL_PATH)

    print(f"Model type: {type(model)}")

    assert model is not None