import torch
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_PATH = "saved_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model

