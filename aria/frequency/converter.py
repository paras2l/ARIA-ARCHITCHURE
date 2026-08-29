"""
ARIA — Unknown Word Handler & Frequency Converter
===================================================
Handles new, out-of-vocabulary words arriving into ARIA dynamically.

1. If context words are available and registered in Guard Layer:
   Calculates initial centroid position from context words + small dispersion noise (±0.05).
2. If no context words are known or available:
   Assigns clean pseudo-random coordinates.
3. Automatically registers the new word in Guard Layer so it is immediately active.
4. Position refines dynamically through the Resonance Loop (gravitational drift).
"""

import random
import string
from aria.guard.layer import exists, get_frequency, auto_register, _generate_seed_coordinate

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset = frozenset({
    "a", "an", "the",
    "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with",
    "and", "or", "but", "not", "nor",
    "it", "its", "this", "that", "these", "those",
    "he", "she", "they", "we", "i", "you",
    "his", "her", "their", "our", "my", "your",
    "as", "by", "from", "up", "about", "into",
    "do", "did", "does", "have", "has", "had",
    "will", "would", "shall", "should", "may", "might",
    "can", "could", "s", "t",
})

_NOISE = 0.05


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def extract_context_words(text: str) -> list[str]:
    """
    Extract meaningful word tokens from raw context text.
    Punctuation stripped, stopwords removed.
    """
    if not text:
        return []
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    cleaned = text.lower().translate(translator)
    return [w for w in cleaned.split() if w and w not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Unknown Word Handler
# ---------------------------------------------------------------------------

def handle_unknown_word(word: str, context_text: str = "") -> dict:
    """
    Handle an incoming word (known or unknown).
    
    If already exists in Guard Layer:
        Returns current coordinates and binary signature.
    If unknown:
        Calculates initial starting position from context centroid + noise,
        or seed coordinates if no context words exist in Guard Layer.
        Auto-registers the new word immediately in Guard Layer.
    """
    word = word.strip()
    if not word:
        return {"word": "", "binary": "", "x": 0.0, "y": 0.0, "z": 0.0, "is_new": False, "based_on": 0}

    # Check if already registered
    if exists(word):
        pos = get_frequency(word)
        return {
            "word": word,
            "binary": auto_register(word)[0],
            "x": pos["x"] if pos else 0.0,
            "y": pos["y"] if pos else 0.0,
            "z": pos["z"] if pos else 0.0,
            "is_new": False,
            "based_on": 0,
        }

    # If context text is provided, try computing context centroid
    context_words = extract_context_words(context_text)
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    used = 0

    for ctx in context_words:
        if ctx != word.lower() and exists(ctx):
            pos = get_frequency(ctx)
            if pos:
                sum_x += pos["x"]
                sum_y += pos["y"]
                sum_z += pos["z"]
                used += 1

    if used > 0:
        avg_x = sum_x / used
        avg_y = sum_y / used
        avg_z = sum_z / used
        rx = round(avg_x + random.uniform(-_NOISE, _NOISE), 6)
        ry = round(avg_y + random.uniform(-_NOISE, _NOISE), 6)
        rz = round(avg_z + random.uniform(-_NOISE, _NOISE), 6)
    else:
        rx, ry, rz = _generate_seed_coordinate(word)

    binary_sig, is_new = auto_register(word, rx, ry, rz)

    return {
        "word": word,
        "binary": binary_sig,
        "x": rx,
        "y": ry,
        "z": rz,
        "is_new": is_new,
        "based_on": used,
    }


def get_rough_position(word: str, context_text: str) -> dict:
    """Backward-compatible helper returning rough position dict."""
    res = handle_unknown_word(word, context_text)
    return {
        "word": word,
        "x": res["x"],
        "y": res["y"],
        "z": res["z"],
        "based_on": res["based_on"],
        "note": "rough position — auto-registered in Guard Layer",
    }


def batch_get_rough_positions(words_with_context: list[dict]) -> list[dict]:
    """Process a batch of words with context dicts."""
    results = []
    for item in words_with_context:
        word = item.get("word", "")
        ctx = item.get("context_text", "")
        if word:
            results.append(get_rough_position(word, ctx))
    return results
