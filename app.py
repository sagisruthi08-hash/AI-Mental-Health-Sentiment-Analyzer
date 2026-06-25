"""
AI Mental Health Sentiment Analyzer
Main Streamlit application entry point.

IMPORTANT: This app provides general wellbeing support and sentiment
tracking. It is not a diagnostic tool and is not a substitute for
professional mental healthcare.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

import database as db
from sentiment_analyzer import analyze_text, detect_crisis_signal, get_mood_bands
from chatbot import ask_support_bot, QUICK_PROMPTS, DISCLAIMER, CRISIS_RESOURCES

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="AI Mental Health Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
)

# -------------------- INIT DB --------------------
db.init_db()

# -------------------- SESSION STATE --------------------
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# -------------------- SIDEBAR --------------------
st.sidebar.title("🧠 Mental Health Sentiment Analyzer")
st.sidebar.caption("Track your mood, journal your thoughts, and get supportive guidance.")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "📝 Journal Check-In",
        "📈 Mood Dashboard",
        "💬 AI Support Chat",
        "📚 Resources & Self-Care",
    ],
)

st.sidebar.markdown("---")
st.sidebar.warning(
    "⚠️ Not a substitute for professional care. If you're in crisis, see the "
    "Resources page or call a helpline immediately."
)

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "i",
    "to", "of", "in", "on", "for", "it", "that", "this", "my", "me", "with",
    "as", "at", "be", "have", "had", "has", "so", "just", "im", "you",
    "your", "not", "no", "do", "did", "about", "they", "we", "feel",
    "feeling", "felt", "today", "really", "very", "all", "am", "if",
}


def _top_words(entries_df: pd.DataFrame, n: int = 10):
    if entries_df.empty:
        return pd.DataFrame(columns=["word", "count"])
    words = []
    for text in entries_df["entry_text"]:
        for w in text.lower().split():
            w = "".join(ch for ch in w if ch.isalpha())
            if len(w) > 2 and w not in _STOPWORDS:
                words.append(w)
    counts = Counter(words).most_common(n)
    return pd.DataFrame(counts, columns=["word", "count"])


# ======================================================
# HOME PAGE
# ======================================================
if page == "🏠 Home":
    st.title("🧠 AI Mental Health Sentiment Analyzer")
    st.subheader("Understand your mood patterns and get supportive guidance")
    st.write(
        "This app helps you reflect on your wellbeing through journaling, sentiment "
        "tracking, and a supportive AI chat companion."
    )
    st.info(DISCLAIMER)

    entries_df = db.get_all_entries()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Journal Entries Logged", len(entries_df))
    with col2:
        st.metric("Check-In Streak", f"{db.get_checkin_streak()} day(s)")
    with col3:
        weekly_avg = db.get_weekly_average_sentiment()
        st.metric("7-Day Avg Sentiment", weekly_avg)

    st.markdown("### Key Features")
    st.markdown(
        "- **Journal Check-In** — write freely; sentiment is analyzed automatically\n"
        "- **Mood Dashboard** — visualize sentiment trends and patterns over time\n"
        "- **AI Support Chat** — a rule-based companion for coping strategies and active listening\n"
        "- **Resources & Self-Care** — helpline numbers and grounding techniques"
    )

# ======================================================
# JOURNAL CHECK-IN PAGE
# ======================================================
elif page == "📝 Journal Check-In":
    st.title("📝 Journal Check-In")
    st.write("Write about how you're feeling. Your entry will be analyzed for sentiment and saved privately.")

    entry_text = st.text_area("What's on your mind today?", height=180)

    if st.button("Analyze & Save Entry") and entry_text.strip():
        result = analyze_text(entry_text)
        crisis = detect_crisis_signal(entry_text)
        db.log_entry(entry_text, result, crisis)

        if crisis:
            st.error(CRISIS_RESOURCES)
        else:
            st.success(f"{result['emoji']} Mood detected: **{result['label']}**  (sentiment score: {result['compound']:.2f})")
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive", f"{result['pos']*100:.0f}%")
            col2.metric("Neutral", f"{result['neu']*100:.0f}%")
            col3.metric("Negative", f"{result['neg']*100:.0f}%")

    st.markdown("---")
    st.markdown("### Mood Scale Reference")
    bands = get_mood_bands()
    st.write("  ".join(f"{emoji} {label}" for label, emoji in bands))

    st.markdown("### Recent Entries")
    recent = db.get_all_entries(limit=5)
    if not recent.empty:
        for _, row in recent.iterrows():
            with st.expander(f"{row['timestamp'].strftime('%b %d, %Y %I:%M %p')} — {row['mood_label']}"):
                st.write(row["entry_text"])
    else:
        st.caption("No entries yet — write your first check-in above.")

# ======================================================
# MOOD DASHBOARD PAGE
# ======================================================
elif page == "📈 Mood Dashboard":
    st.title("📈 Mood Dashboard")

    entries_df = db.get_all_entries()

    if entries_df.empty:
        st.warning("No journal entries yet. Add a few check-ins to see your trends here.")
    else:
        st.markdown("### Sentiment Over Time")
        trend_df = entries_df.sort_values("timestamp")
        fig = px.line(
            trend_df, x="timestamp", y="compound", markers=True,
            labels={"timestamp": "Date", "compound": "Sentiment Score"},
            title="Sentiment Score Trend",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Mood Distribution")
            mood_counts = entries_df["mood_label"].value_counts().reset_index()
            mood_counts.columns = ["mood_label", "count"]
            fig2 = px.pie(mood_counts, names="mood_label", values="count", title="Mood Breakdown")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("### Most Frequent Words")
            words_df = _top_words(entries_df)
            if not words_df.empty:
                fig3 = px.bar(words_df, x="word", y="count", title="Top Words in Your Entries")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.caption("Not enough text yet to surface common words.")

        st.markdown("### Entry Log")
        display_df = entries_df[["timestamp", "mood_label", "compound", "entry_text"]].copy()
        display_df.columns = ["Date", "Mood", "Score", "Entry"]
        st.dataframe(display_df, use_container_width=True)

# ======================================================
# AI SUPPORT CHAT PAGE
# ======================================================
elif page == "💬 AI Support Chat":
    st.title("💬 AI Support Chat")
    st.info(DISCLAIMER)

    st.markdown("**Quick Prompts:**")
    cols = st.columns(len(QUICK_PROMPTS))
    for i, q in enumerate(QUICK_PROMPTS):
        if cols[i].button(q, key=f"quick_{i}"):
            st.session_state.chat_messages.append({"role": "user", "content": q})
            response, crisis = ask_support_bot(q, st.session_state.chat_messages[:-1])
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            db.log_chat(q, response, crisis)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Share what's on your mind...")
    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        response, crisis = ask_support_bot(user_input, st.session_state.chat_messages[:-1])
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
        db.log_chat(user_input, response, crisis)

# ======================================================
# RESOURCES & SELF-CARE PAGE
# ======================================================
elif page == "📚 Resources & Self-Care":
    st.title("📚 Resources & Self-Care")

    st.error(
        "**If you are in immediate danger or having thoughts of suicide or self-harm, "
        "please contact emergency services or a crisis helpline right now.**"
    )
    st.markdown(CRISIS_RESOURCES)

    st.markdown("---")
    st.markdown("### Grounding & Coping Techniques")
    st.markdown(
        "- **Box breathing:** inhale 4s → hold 4s → exhale 4s → hold 4s, repeat for a few minutes\n"
        "- **5-4-3-2-1 grounding:** name 5 things you see, 4 you hear, 3 you can touch, 2 you smell, 1 you taste\n"
        "- **Brief movement:** a short walk or stretch can shift a stuck mental state\n"
        "- **Reach out:** a quick message to a friend or family member can ease isolation\n"
        "- **Write it down:** journaling (like the Check-In page) can help externalize racing thoughts"
    )

    st.markdown("### When to Seek Professional Support")
    st.write(
        "Consider speaking with a doctor, counselor, or therapist if low mood, anxiety, or "
        "stress are persistent, are affecting your daily life, or simply if you want support — "
        "you don't need to be in crisis to benefit from professional care."
    )

    st.caption(
        "This app is a self-reflection and support tool, not a diagnostic or clinical "
        "service. Sentiment scores reflect language patterns only, not a medical assessment."
    )