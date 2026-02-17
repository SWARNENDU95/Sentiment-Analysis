from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from model import load_model

app = FastAPI()

tokenizer, model = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: TextInput):
    inputs = tokenizer(
        data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    confidence = torch.max(probs).item()
    prediction = torch.argmax(probs).item()

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "sentiment": sentiment,
        "confidence": round(confidence, 4)
    }

