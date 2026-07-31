import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text, keyword_enhance, get_mixed_emotions, EMOTION_LABELS

MAX_SEQ_LEN = 80
NUM_CLASSES = 5
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'bltsm')


def focal_loss(gamma=2.0):
    """
    Focal Loss for class imbalance (gamma=2.0, as used during Kaggle training).
    Must be provided as a custom_object when loading the .keras model.
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_true_one_hot = tf.one_hot(y_true, depth=NUM_CLASSES)
        ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal = tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(focal)
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def load_bilstm_model():
    """Load BiLSTM model, tokenizer, and label classes from disk."""
    model_path = os.path.join(MODEL_DIR, 'bilstm_student_adaptive.keras')
    tokenizer_path = os.path.join(MODEL_DIR, 'tokenizer.pkl')
    labels_path = os.path.join(MODEL_DIR, 'label_classes.npy')

    # focal_loss must be passed as a custom_object because the model was trained with it.
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'loss_fn': focal_loss(gamma=2.0)}
    )

    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    label_classes = np.load(labels_path, allow_pickle=True)
    return model, tokenizer, label_classes


def predict_bilstm(raw_text: str, model, tokenizer, label_classes) -> dict:
    """
    Run BiLSTM inference on student input text.

    Pipeline:
    1. Clean text
    2. Tokenize and pad to MAX_SEQ_LEN=80
    3. Model inference → softmax probabilities
    4. Keyword enhancement (10x boost)
    5. Build unified result schema

    Returns dict with keys: emotion, confidence, scores, cleaned_text
    """
    cleaned = clean_text(raw_text)
    sequences = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    raw_probs = model.predict(padded, verbose=0)[0]
    # Apply softmax normalization
    exp_probs = np.exp(raw_probs - np.max(raw_probs))
    probs = exp_probs / exp_probs.sum()

    # Keyword enhancement
    enhanced = keyword_enhance(raw_text, probs)

    scores = {label_classes[i]: float(enhanced[i]) for i in range(len(label_classes))}
    top_emotion = max(scores, key=scores.get)
    confidence = scores[top_emotion]

    return {
        'emotion': top_emotion,
        'confidence': confidence,
        'scores': scores,
        'cleaned_text': cleaned
    }
