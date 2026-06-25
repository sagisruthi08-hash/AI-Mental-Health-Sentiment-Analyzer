"""
sentiment_analyzer.py
Lexicon-based sentiment analysis (VADER) for journal entries and chat
messages, plus a conservative crisis-signal detector used to surface
helpline resources when needed.

This module never produces a clinical diagnosis — it only reports
sentiment polarity (negative/neutral/positive) and intensity.
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Mood label bands, keyed by VADER's compound score (-1 to 1).
# These are sentiment-intensity labels, not clinical/diagnostic terms.
# The last entry acts as the guaranteed fallback (no threshold needed).
_MOOD_BANDS = [
    (0.5,   "Very Positive", "😄"),
    (0.15,  "Positive",      "🙂"),
    (-0.15, "Neutral",       "😐"),
    (-0.5,  "Low",           "😔"),
    # FIX 1 & 2: Replaced magic -1.01 sentinel with float('-inf') so the
    # last band is always reached explicitly, not by accident.
    (float("-inf"), "Very Low", "😢"),
]

# Conservative keyword set used only to decide whether to surface crisis
# resources. Detection is intentionally broad/cautious (better a false
# positive showing helpline info than a missed signal).
#
# FIX 5: Expanded variants — "killing myself", "kill my self", etc. —
# and removed exact duplicates that regex alternation already covers.
_CRISIS_PHRASES = [
    r"kill(?:ing)?\s+my\s*self",
    r"suicide",
    r"suicidal",
    r"end\s+my\s+life",
    r"end\s+it\s+all",
    r"want\s+to\s+die",
    r"don'?t\s+want\s+to\s+live",
    r"no\s+reason\s+to\s+live",
    r"self[\s\-]harm(?:ing)?",
    r"hurt(?:ing)?\s+my\s*self",
    r"cut(?:ting)?\s+my\s*self",
    r"better\s+off\s+dead",
    r"can'?t\s+go\s+on(?:\s+living)?",
]

# FIX 4: Pre-compile a single word-boundary-aware pattern so substring
# false-positives (e.g. "shelf harm") are avoided.
_CRISIS_PATTERN = re.compile(
    r"\b(?:" + "|".join(_CRISIS_PHRASES) + r")\b",
    re.IGNORECASE,
)


def analyze_text(text: str) -> dict:
    """
    Run sentiment analysis on a piece of text.

    Returns a dict with keys:
        compound (float | None), pos, neu, neg, label (str), emoji (str)

    FIX 3 & 6: Returns label="Unknown" for non-string or empty input
    instead of silently treating it as neutral sentiment.
    """
    # Guard: reject non-string and blank input explicitly.
    if not isinstance(text, str) or not text.strip():
        return {
            "compound": None,
            "pos":      None,
            "neu":      None,
            "neg":      None,
            "label":    "Unknown",
            "emoji":    "❓",
        }

    scores   = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    # FIX 1 & 2: Loop uses float('-inf') as the last threshold, so the
    # correct band is always matched explicitly — no accidental fall-through.
    label, emoji = "Very Low", "😢"          # safe default (never used)
    for threshold, band_label, band_emoji in _MOOD_BANDS:
        if compound >= threshold:
            label, emoji = band_label, band_emoji
            break

    return {
        "compound": compound,
        "pos":      scores["pos"],
        "neu":      scores["neu"],
        "neg":      scores["neg"],
        "label":    label,
        "emoji":    emoji,
    }


def detect_crisis_signal(text: str) -> bool:
    """
    Conservative pattern check for crisis language. This is NOT a
    diagnostic tool and will both miss things and over-trigger — it
    exists only to decide when to proactively surface helpline info.

    FIX 4 & 6: Uses word-boundary regex instead of naive substring
    matching, and guards against non-string input.
    """
    if not isinstance(text, str) or not text:
        return False
    return bool(_CRISIS_PATTERN.search(text))


def get_mood_bands() -> list[tuple[str, str]]:
    """Expose the band definitions (used for legends/labels in the UI)."""
    return [(label, emoji) for _, label, emoji in _MOOD_BANDS]