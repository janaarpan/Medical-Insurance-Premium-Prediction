from fastapi import FastAPI, HTTPException
from model.predict import MODEL_VERSION, predict_insurance_premium
from schemas.user_input import UserInput
from schemas.output_validation import PredictionResponse

app = FastAPI(
    title="Medical Insurance Premium Prediction API",
    description="Predicts an insurance premium category using applicant attributes.",
    version=MODEL_VERSION,
)

@app.get("/", tags=["General"])
def root():
    return {"message": "Medical Insurance Premium Prediction API", "docs": "/docs"}

@app.get("/health", tags=["General"])
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}

@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict an insurance premium category",
)
def predict(payload: UserInput) -> PredictionResponse:
    model_features = {
        "bmi": payload.bmi,
        "age_group": payload.age_group,
        "lifestyle_risk": payload.lifestyle_risk,
        "city_tier": payload.city_tier,
        "income_lpa": payload.income_lpa,
        "occupation": payload.occupation,
    }
    try:
        return predict_insurance_premium(model_features)
    except Exception as exc:
        # Log the exception in production; do not expose internal details to clients.
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.") from exc
