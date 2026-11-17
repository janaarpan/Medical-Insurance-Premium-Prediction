from pathlib import Path
import pickle
import pandas as pd

MODEL_PATH = Path(__file__).resolve().with_name("model.pkl")

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

MODEL_VERSION = "1.0.0"

def predict_insurance_premium(user_input: dict) -> dict:
    """Return the predicted premium category and class probabilities."""
    input_frame = pd.DataFrame([user_input])
    predicted_category = str(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    class_probabilities = {
        str(label): round(float(probability), 4)
        for label, probability in zip(model.classes_, probabilities)
    }
    return {
        "predicted_category": predicted_category,
        "confidence": round(float(max(probabilities)), 4),
        "class_probabilities": class_probabilities,
    }
