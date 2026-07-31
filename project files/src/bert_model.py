import numpy as np
import json
import os
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
from src.preprocessing import clean_text, keyword_enhance, EMOTION_LABELS

BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'bert_emotion_model_final')

# Class weights tuned during training: [Bored, Confident, Confused, Curious, Frustrated]
CLASS_WEIGHTS = [1.2, 1.8, 0.6, 1.0, 1.4]

# Keywords that trigger additional confidence/confusion boost in BERT predictions
CONFIDENCE_KEYWORDS = ['comfortable', 'confident', 'easy', 'clear', 'understand', 'got it', 'makes sense']
CONFUSION_KEYWORDS = ['confused', 'unclear', 'lost', "don't understand", 'puzzled']


def load_bert_model():
    """Load fine-tuned BERT model, tokenizer, and label mapping from disk."""
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
    model.eval()

    label_path = os.path.join(BERT_MODEL_DIR, 'label_mapping.json')
    with open(label_path, 'r') as f:
        label_mapping = json.load(f)  # e.g. {"0": "Bored", "1": "Confident", ...}

    return model, tokenizer, label_mapping


def predict_bert(raw_text: str, model, tokenizer, label_mapping) -> dict:
    """
    Run BERT inference on student input text.

    Pipeline:
    1. Clean text
    2. BERT tokenization (max_length=128, truncation, padding)
    3. Forward pass → logits → softmax probabilities
    4. Apply class weights [1.2, 1.8, 0.6, 1.0, 1.4]
    5. Keyword boost: Confident class * 2.5 if confidence keywords found
                      Confused class * 2.0 if confusion keywords found
    6. Renormalize
    7. Build unified result schema

    Returns dict with keys: emotion, confidence, scores, cleaned_text
    """
    cleaned = clean_text(raw_text)
    inputs = tokenizer(
        cleaned,
        return_tensors='pt',
        max_length=128,
        truncation=True,
        padding='max_length'
    )

    with torch.no_grad():
        logits = model(**inputs).logits[0].numpy()

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    # Apply class weights
    weighted = probs * np.array(CLASS_WEIGHTS)
    weighted = weighted / weighted.sum()

    # Keyword adjustments
    text_lower = raw_text.lower()
    if any(kw in text_lower for kw in CONFIDENCE_KEYWORDS):
        # Boost Confident (index 1) and Reduce Confused (index 2)
        confident_idx = list(label_mapping.values()).index('Confident')
        confused_idx = list(label_mapping.values()).index('Confused')
        weighted[confident_idx] *= 2.5
        weighted[confused_idx] *= 0.3
    elif any(kw in text_lower for kw in CONFUSION_KEYWORDS):
        # Boost Confused (index 2)
        confused_idx = list(label_mapping.values()).index('Confused')
        weighted[confused_idx] *= 2.0

    weighted = weighted / weighted.sum()

    scores = {label_mapping[str(i)]: float(weighted[i]) for i in range(len(label_mapping))}
    top_emotion = max(scores, key=scores.get)
    confidence = scores[top_emotion]

    return {
        'emotion': top_emotion,
        'confidence': confidence,
        'scores': scores,
        'cleaned_text': cleaned
    }
