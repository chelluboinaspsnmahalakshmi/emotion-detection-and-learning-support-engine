"""
src/train.py — Local training utility (mirrors the Kaggle notebook logic).

This script is a reference implementation for re-training the BiLSTM model locally.
The primary training environment is Kaggle (GPU), documented in:
  notebooks/kaggle_training.ipynb

To retrain locally (CPU only, slower):
  python -m src.train

Outputs saved to:
  models/bltsm/bilstm_student_adaptive.keras
  models/bltsm/tokenizer.pkl
  models/bltsm/label_classes.npy
"""

import os
import re
import time
import pickle
import random
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ─── Hyperparameters (must match Kaggle training exactly) ─────────────────────
MAX_VOCAB_SIZE = 30_000
MAX_SEQ_LEN    = 80
EMBED_DIM      = 128       # Embedding layer dimension
LSTM_UNITS     = 128       # Bidirectional LSTM units (128 × 2 = 256 effective)
NUM_CLASSES    = 5

# Training phase 1 — baseline BiLSTM
EPOCHS_BASELINE    = 10
BATCH_SIZE_BASELINE = 512  # Large batch for GPU efficiency

# Training phase 2 — domain-adaptive fine-tuning
EPOCHS_FINETUNE    = 8
BATCH_SIZE_FINETUNE = 64
STUDENT_SAMPLES_PER_CLASS = 2000  # → 10,000 total student samples

TARGET_EMOTIONS = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'bltsm')
DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(MODEL_DIR, exist_ok=True)

STOPWORDS = set(stopwords.words('english'))


# ─── Focal Loss (γ=2.0) ───────────────────────────────────────────────────────
def focal_loss(gamma: float = 2.0):
    """
    Focal Loss for class imbalance.
    gamma=2.0 matches Kaggle training config.
    Must be registered as a custom_object when loading the saved .keras file.
    """
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.int32)
        y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_ohe   = tf.one_hot(y_true, depth=NUM_CLASSES)
        ce      = -tf.reduce_sum(y_ohe * tf.math.log(y_pred), axis=-1)
        pt      = tf.reduce_sum(y_ohe * y_pred, axis=-1)
        return tf.reduce_mean(tf.pow(1.0 - pt, gamma) * ce)
    loss_fn.__name__ = 'focal_loss'
    return loss_fn


# ─── Text Cleaning (identical to preprocessing.py & Kaggle Cell 3) ───────────
def clean_text(text: str) -> str:
    """Normalize text while preserving emotion-carrying punctuation."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)   # remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)       # keep only alpha
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)


# ─── Synthetic Student-Domain Data ───────────────────────────────────────────
def generate_student_data(n_per_class: int = STUDENT_SAMPLES_PER_CLASS) -> pd.DataFrame:
    """
    Generate synthetic student-specific training sentences for fine-tuning.
    Mirrors Cell 5 of notebooks/kaggle_training.ipynb.
    Total: n_per_class × 5 = 10,000 samples by default.
    """
    templates = {
        'Bored': [
            "This topic is so {0} and {1}",
            "I find {0} really {1} and unengaging",
            "{0} is the most boring subject I have ever studied",
        ],
        'Confident': [
            "I understand {0} perfectly now",
            "I've mastered {0} and feel great about {1}",
            "Finally {0} makes sense! I've got it.",
        ],
        'Confused': [
            "I don't get {0} at all",
            "Can someone explain {0} to me?",
            "I keep getting lost whenever we cover {0}",
        ],
        'Curious': [
            "I wonder how {0} works exactly",
            "I'm fascinated by {0} and want to learn more about {1}",
            "Why does {0} behave this way? I really want to understand",
        ],
        'Frustrated': [
            "I can't figure out {0} no matter what",
            "This {0} is impossible and I'm giving up on {1}",
            "I've tried {0} ten times and I still can't get it right",
        ],
    }
    topics = [
        "recursion", "calculus", "neural networks", "linear algebra",
        "sorting algorithms", "probability", "thermodynamics", "chemical bonds",
        "photosynthesis", "supply curves", "pointers", "integrals", "Bayes theorem",
        "eigenvectors", "gradient descent", "backpropagation", "Fourier transforms",
    ]
    random.seed(42)
    rows = []
    for emotion, tmpl_list in templates.items():
        for _ in range(n_per_class):
            tmpl = random.choice(tmpl_list)
            try:
                text = tmpl.format(random.choice(topics), random.choice(topics))
            except IndexError:
                text = tmpl.format(random.choice(topics))
            rows.append({'text': text, 'emotion': emotion})
    return pd.DataFrame(rows)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load emotion_text_dataset.csv (exported from Kaggle after Cell 2).
    Falls back to pure synthetic data if file is not present.
    """
    csv_path = os.path.join(DATA_DIR, 'emotion_text_dataset.csv')
    if os.path.exists(csv_path):
        print(f"✅ Loading combined dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        df = df[df['emotion'].isin(TARGET_EMOTIONS)].dropna(subset=['text', 'emotion'])
        print(f"📊 Loaded {len(df):,} samples from CSV")
    else:
        print("⚠️  data/emotion_text_dataset.csv not found.")
        print("   Tip: Run notebooks/kaggle_training.ipynb on Kaggle to generate it.")
        print("   Falling back to synthetic data only (limited accuracy expected).\n")
        df = pd.DataFrame({'text': [], 'emotion': []})

    # Always augment with student-domain synthetic samples
    student_df = generate_student_data()
    df = pd.concat([df, student_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"📊 Total samples after augmentation: {len(df):,}")
    print(df['emotion'].value_counts().to_string())
    return df


# ─── Build BiLSTM Model ───────────────────────────────────────────────────────
def build_bilstm_model() -> tf.keras.Model:
    """
    Exact BiLSTM architecture from Epic 2 (4.1M parameters):
      Embedding(30000, 128, mask_zero=True)
      → Bidirectional(LSTM(128, dropout=0.2, use_cudnn=False))   # cuDNN mask fix
      → Dense(128, relu)
      → Dropout(0.3)
      → Dense(5, softmax)
    """
    model = Sequential([
        Embedding(MAX_VOCAB_SIZE, EMBED_DIM, input_length=MAX_SEQ_LEN, mask_zero=True),
        Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, use_cudnn=False)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax'),
    ], name='bilstm_baseline')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy'],
    )
    return model


# ─── Main Training Pipeline ───────────────────────────────────────────────────
def main():
    print("🚀 BiLSTM Training Pipeline")
    print("   Primary GPU training → notebooks/kaggle_training.ipynb")
    print("   This script = local CPU fallback / re-training reference\n")

    # ── 1. Load & clean data ──────────────────────────────────────────────────
    df = load_data()
    print("\n🧹 Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)

    # ── 2. Tokenization ───────────────────────────────────────────────────────
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(df['clean_text'])
    sequences = tokenizer.texts_to_sequences(df['clean_text'])
    padded = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    print(f"✅ Tokenization complete: {padded.shape}")

    # ── 3. Label encoding ─────────────────────────────────────────────────────
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['emotion'])
    label_classes = label_encoder.classes_
    print(f"📋 Classes: {list(label_classes)}")

    # ── 4. Train / Val / Test split (80 / 10 / 10) ───────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        padded, y, test_size=0.10, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
    )
    print(f"📦 Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── 5. Build & train baseline BiLSTM ─────────────────────────────────────
    baseline_model = build_bilstm_model()
    baseline_model.summary()

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2, factor=0.5),          # matches training screenshot
    ]

    print(f"\n🚀 Training (balanced + focal)...")
    start = time.time()
    baseline_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_BASELINE,
        batch_size=BATCH_SIZE_BASELINE,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\n⏱ Training time: {(time.time() - start)/60:.2f} min")

    # ── 6. Evaluate baseline ──────────────────────────────────────────────────
    y_pred = np.argmax(baseline_model.predict(X_test, verbose=0), axis=1)
    print("\n📊 CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, target_names=label_classes, digits=4))
    print("Prediction distribution:")
    print(pd.Series(y_pred).value_counts(normalize=True).to_string())

    # ── 7. Domain-Adaptive Fine-Tuning ────────────────────────────────────────
    # Clone baseline, copy weights, freeze embedding layer (layer[0])
    print("\n🎯 Cloning model for domain adaptation...")
    adaptive_model = tf.keras.models.clone_model(baseline_model)
    adaptive_model.set_weights(baseline_model.get_weights())
    adaptive_model.layers[0].trainable = False          # freeze Embedding

    adaptive_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',          # standard CE for fine-tuning
        metrics=['accuracy'],
    )
    print("🟢 Adaptive model ready")

    # Generate 10,000 student-specific samples (2,000 × 5 classes)
    student_df = generate_student_data(n_per_class=STUDENT_SAMPLES_PER_CLASS)
    student_df['clean_text'] = student_df['text'].apply(clean_text)
    s_seqs   = tokenizer.texts_to_sequences(student_df['clean_text'])
    s_padded = pad_sequences(s_seqs, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    s_y      = label_encoder.transform(student_df['emotion'])

    s_X_train, s_X_val, s_y_train, s_y_val = train_test_split(
        s_padded, s_y, test_size=0.1, random_state=42, stratify=s_y
    )

    ft_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ]

    print("\n🚀 Fine-tuning on student domain...")
    adaptive_model.fit(
        s_X_train, s_y_train,
        validation_data=(s_X_val, s_y_val),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE_FINETUNE,
        callbacks=ft_callbacks,
        verbose=1,
    )

    val_loss, val_acc = adaptive_model.evaluate(s_X_val, s_y_val, verbose=0)
    print(f"\n🎯 Student Adaptive Validation Accuracy: {val_acc:.4f}")

    # ── 8. Save all artifacts ─────────────────────────────────────────────────
    adaptive_model.save(os.path.join(MODEL_DIR, 'bilstm_student_adaptive.keras'))
    with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    np.save(os.path.join(MODEL_DIR, 'label_classes.npy'), label_classes)

    print(f"\n✅ Artifacts saved to {MODEL_DIR}/")
    print("   ├── bilstm_student_adaptive.keras")
    print("   ├── tokenizer.pkl")
    print("   └── label_classes.npy")
    print("\n📋 Next steps:")
    print("   1. Copy these files to your local models/bltsm/ folder")
    print("   2. Run the BERT fine-tuning cells in the Kaggle notebook")
    print("   3. Copy bert_emotion_model_final/ to your local models/ folder")


if __name__ == '__main__':
    main()
