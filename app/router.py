from fastapi import APIRouter
from pydantic import BaseModel
from src.decision_layer import AIDecisionLayer
from src.config import MODELS_DIR

router = APIRouter()

ai = AIDecisionLayer(
    intent_model_path=str(MODELS_DIR / "intent_model.joblib"),
    demand_model_path=str(MODELS_DIR / "demand_model.joblib"),
)
ai.load_models()


class ChatRequest(BaseModel):
    message: str


@router.post("/chat")
def chat(req: ChatRequest):
    result = ai.handle(req.message)
    return result
