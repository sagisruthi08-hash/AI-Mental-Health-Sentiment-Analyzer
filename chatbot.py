"""
chatbot.py
A lightweight, rule-based supportive chat assistant. No external API
calls — this is a wellbeing companion that offers coping suggestions
and active listening prompts. It is NOT a therapist and does not
diagnose; when crisis language is detected, it leads with helpline
resources instead of generic advice.
"""

from sentiment_analyzer import detect_crisis_signal

QUICK_PROMPTS = [
    "I'm feeling anxious",
    "I'm stressed about work/school",
    "I had a really good day",
    "I'm feeling lonely",
]

DISCLAIMER = (
    "This chat offers general wellbeing support and is not a substitute "
    "for professional mental healthcare. If you're in crisis, please "
    "reach out to a helpline or trusted person — see below."
)

# Helplines verified as current. India: KIRAN and Tele-MANAS are
# government-run, 24/7, multi-language national lines. Add/adjust
# local numbers as needed for your audience.
CRISIS_RESOURCES = (
    "**You don't have to go through this alone.** Please reach out right now:\n\n"
    "- 🇮🇳 **KIRAN Helpline (India):** 1800-599-0019 (24/7, toll-free, multilingual)\n"
    "- 🇮🇳 **Tele-MANAS (India):** 14416 or 1-800-891-4416 (24/7)\n"
    "- 🌍 **Befrienders Worldwide:** befrienders.org — find a crisis line in your country\n"
    "- If you are in immediate danger, please contact local emergency services or go to "
    "the nearest emergency room.\n\n"
    "Talking to a mental health professional, a doctor, or someone you trust can make "
    "a real difference. Would you like to tell me a bit more about how you're feeling?"
)

_RULES = [
    (["anxious", "anxiety", "panic", "worried", "nervous"],
     "Anxiety can feel overwhelming. One thing that helps in the moment is slow "
     "breathing — try inhaling for 4 seconds, holding for 4, and exhaling for 6. "
     "Would you like to share what's making you feel anxious right now?"),

    (["stress", "stressed", "overwhelmed", "pressure", "deadline"],
     "It sounds like you have a lot on your plate. Breaking things into smaller, "
     "specific next steps can make a workload feel more manageable. What's the one "
     "thing that's weighing on you most right now?"),

    (["sad", "down", "low", "upset", "crying"],
     "I'm sorry you're feeling this way — that sounds hard. Sometimes naming what's "
     "going on can help. Is there something specific that brought this on today?"),

    (["lonely", "alone", "isolated", "no one"],
     "Feeling lonely is genuinely painful, even when people are around. Is there "
     "someone — a friend, family member, or counselor — you could reach out to "
     "today, even just for a short chat?"),

    (["angry", "frustrated", "mad", "annoyed"],
     "That frustration sounds valid. Sometimes stepping away for a few minutes, "
     "or writing down exactly what's bothering you, can help take the edge off. "
     "What happened?"),

    (["tired", "exhausted", "burnt out", "burned out", "no energy"],
     "Running on empty makes everything harder. Even small rest — a short walk, "
     "a few minutes away from screens, an early night — can help rebuild some "
     "energy. How has your sleep been lately?"),

    (["good day", "happy", "great", "grateful", "excited", "proud"],
     "That's wonderful to hear! It's worth pausing to notice what made today good "
     "— what was the highlight?"),
]

_FALLBACK = (
    "Thanks for sharing that. I'm here to listen — feel free to tell me more about "
    "what's on your mind, or try one of the quick prompts above."
)


def ask_support_bot(message: str, history: list = None) -> tuple:
    """
    Return (response_text, crisis_flag) for the given message.
    `history` is accepted for API compatibility with a conversational UI.
    """
    if detect_crisis_signal(message):
        return CRISIS_RESOURCES, True

    lowered = (message or "").lower()
    for keywords, response in _RULES:
        if any(kw in lowered for kw in keywords):
            return response, False

    return _FALLBACK, False