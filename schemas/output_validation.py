from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    predicted_category: str = Field(description="Predicted premium category")
    confidence: float = Field(ge=0, le=1, description="Probability of predicted category")
    class_probabilities: dict[str, float] = Field(description="Probability for each class")
