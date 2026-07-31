import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# Keyword lists for rule-based emotion boosting
EMOTION_KEYWORDS = {
    'Frustrated': ['frustrated', 'frustrating', 'annoying', 'angry', 'hate', 'difficult', 'stuck', 'wrong answer', 'keep getting', 'unnecessarily complicated', 'tried'],
    'Curious': ['why', 'how', 'what', 'curious', 'wonder', 'interested', 'learn', 'know more', 'want to know', 'explore', 'could we', 'what happens', 'intuition', 'beh'],
    'Confident': ['easy', 'amazing', 'great', 'excellent', 'good', 'awesome', 'perfect', 'solved', 'got it', 'clear now', 'finally', 'move ahead', 'understand clearly'],
    'Bored': ['boring', 'bored', 'tired', 'repetitive', 'dull', 'not engaging', 'didnt feel engaging', "didn't feel engaging", 'not interesting', 'too basic', 'losing'],
    'Confused': ['confused', 'lost', 'unclear', 'dont understand', "don't understand", "doesn't make sense", 'not fully confident', 'missing', 'incomplete', 'unsure']
}

EMOTION_LABELS = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']


def clean_text(text: str) -> str:
    """Clean and normalize student input text while preserving emotional punctuation."""
    text = str(text).lower()
    # Keep punctuation that indicates emotion
    text = re.sub(r'[^a-zA-Z\s,!]', ' ', text)
    tokens = word_tokenize(text)
    
    # Keep ALL meaningful words, remove only basic articles
    skip_words = {'the', 'a', 'an'}
    tokens = [t for t in tokens if t not in skip_words and len(t) > 1]
    
    return ' '.join(tokens) if tokens else text


def keyword_enhance(raw_text: str, scores: np.ndarray) -> np.ndarray:
    """
    Apply advanced keyword weight boosting to emotion probability scores.
    """
    text_lower = raw_text.lower()
    probs = np.copy(scores)

    # Score each emotion based on keyword matches with higher weights for explicit mentions
    emotion_scores = {}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in text_lower:
                # Give much higher weight to explicit emotion words
                if keyword in ['frustrated', 'frustrating', 'curious', 'confident', 'bored', 'boring', 'confused']:
                    score += 10 # Very high weight for explicit emotions
                else:
                    score += 2
        emotion_scores[emotion] = score

    max_score = max(emotion_scores.values()) if emotion_scores else 0
    if max_score > 0:
        # Boost the emotion(s) with highest keyword matches
        for emotion, score in emotion_scores.items():
            if score == max_score:
                emotion_idx = EMOTION_LABELS.index(emotion)
                probs[emotion_idx] *= (1 + score * 3.0) # Very strong boost for keyword matches
                
        # Reduce other emotions more aggressively
        winning_emotions = [e for e, s in emotion_scores.items() if s == max_score]
        for i, emotion in enumerate(EMOTION_LABELS):
            if emotion not in winning_emotions and max_score >= 5: # Lower threshold for strong override
                probs[i] *= 0.01 # Very strong reduction

    probs = probs / np.sum(probs) # Renormalize
    return probs


def get_mixed_emotions(scores: dict, threshold: float = 0.15) -> list:
    """
    Return all emotions with score >= threshold, sorted descending.
    Used to detect mixed emotional states.

    Args:
        scores: dict of {emotion_name: score}
        threshold: minimum score to be considered a detected emotion (default 0.15)
    Returns:
        list of (emotion_name, score) tuples for all emotions above threshold
    """
    mixed = [(emotion, score) for emotion, score in scores.items() if score >= threshold]
    mixed.sort(key=lambda x: x[1], reverse=True)
    return mixed
