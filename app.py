import os
import pandas as pd
import streamlit as st
import google.generativeai as genai

from utils.predictor import predict_emotion
from utils.multilabel_bilstm_predictor import predict_multilabel_bilstm
from utils.logger import save_history
from utils.dashboard import (
    emotion_chart,
    emotion_pie_chart,
    confidence_timeline,
    field_chart,
    summary_stats
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Emotion Detection & Learning Support Engine",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "bert_result" not in st.session_state:
    st.session_state.bert_result = None

if "multilabel_result" not in st.session_state:
    st.session_state.multilabel_result = None

if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = ""

# =====================================================
# LOAD HISTORY
# =====================================================

HISTORY_FILE = "history/history.csv"

if (
    len(st.session_state.emotion_history) == 0
    and os.path.exists(HISTORY_FILE)
):
    try:
        history = pd.read_csv(HISTORY_FILE)
        st.session_state.emotion_history = history.to_dict("records")
    except:
        pass

# =====================================================
# GEMINI CONFIG
# =====================================================

genai.configure(
    api_key="GEMINI_API_KEY"  # Replace with your actual Gemini API key
)

gemini_model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =====================================================
# GEMINI FUNCTION
# =====================================================

def get_gemini_response(field, question, emotions):

    prompt = f"""
You are an AI Learning Assistant.

Subject:
{field}

Detected Emotions:
{", ".join(emotions)}

Student Question:
{question}

Provide:

1. Simple explanation
2. Why the student feels these emotions
3. Three study tips
4. One example
5. Motivation

Keep the answer friendly and encouraging.
"""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception as e:
        return str(e)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙ Settings")

use_ai = st.sidebar.checkbox(
    "Use Gemini AI",
    value=True
)

save_csv = st.sidebar.checkbox(
    "Save History",
    value=True
)

show_details = st.sidebar.checkbox(
    "Show Prediction Details",
    value=True
)

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear Session"):

    st.session_state.analysis_done = False
    st.session_state.bert_result = None
    st.session_state.multilabel_result = None
    st.session_state.gemini_response = ""
    st.session_state.emotion_history = []

    st.rerun()

# =====================================================
# HEADER
# =====================================================

st.title("🎓 AI Learning Assistant")

st.markdown("""
Analyze students' learning emotions using

- 🤖 BERT
- 🧠 Multi-label BiLSTM
- 💡 Google Gemini AI

Supports

✅ Single Emotion Detection

✅ Mixed Emotion Detection

✅ AI Learning Guidance

✅ Analytics Dashboard
""")

# =====================================================
# INPUT SECTION
# =====================================================

field = st.selectbox(
    "Select Subject",
    [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Engineering",
        "Business",
        "History",
        "Literature",
        "Other"
    ]
)

question = st.text_area(
    "Describe your learning problem",
    height=180,
    placeholder="Example: I am interested in SQL but normalization is confusing."
)

col1, col2 = st.columns(2)

with col1:
    analyze = st.button(
        "🚀 Analyze",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear:

    st.session_state.analysis_done = False
    st.session_state.bert_result = None
    st.session_state.multilabel_result = None
    st.session_state.gemini_response = ""

    st.rerun()

# =====================================================
# ANALYZE
# =====================================================

if analyze:

    if question.strip() == "":

        st.warning("Please enter your learning problem.")
        st.stop()

    with st.spinner("Analyzing your learning emotion..."):

        try:

            # BERT Prediction
            bert = predict_emotion(question)

            # Multi-label BiLSTM Prediction
            multilabel = predict_multilabel_bilstm(question)

            # Save predictions
            st.session_state.bert_result = bert
            st.session_state.multilabel_result = multilabel
            st.session_state.analysis_done = True

        except Exception as e:

            st.error(e)
            st.stop()

# =====================================================
# LOAD RESULTS
# =====================================================

if st.session_state.analysis_done:

    bert = st.session_state.bert_result
    multilabel = st.session_state.multilabel_result

# =====================================================
# RESULTS
# =====================================================

if st.session_state.analysis_done:

    st.success("✅ Analysis Completed Successfully!")

    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    st.markdown("---")
    st.subheader("🤖 Model Comparison")

    col1, col2 = st.columns(2)

    with col1:

        st.info("### 🤖 BERT")

        st.metric(
            "Primary Emotion",
            bert["emotion"].capitalize(),
            f"{bert['confidence']:.1%}"
        )

        st.write("Secondary Emotion")

        st.write(
            f"**{bert['secondary_emotion'].capitalize()}** "
            f"({bert['secondary_confidence']:.1%})"
        )

    with col2:

        st.info("### 🧠 Multi-label BiLSTM")

        st.metric(
            "Primary Emotion",
            multilabel["emotion"].capitalize(),
            f"{multilabel['confidence']:.1%}"
        )

    st.write("Detected Emotions")

    for item in multilabel["mixed_emotions"]:

        st.write(
            f"• **{item['emotion'].capitalize()}** "
            f"({item['confidence']:.1%})"
        )

    # =====================================================
    # MIXED EMOTION CARDS
    # =====================================================

    st.markdown("---")
    st.subheader("😊 Detected Emotions")

    colors = {
        "Confused": "#F39C12",
        "Curious": "#3498DB",
        "Confident": "#2ECC71",
        "Frustrated": "#E74C3C",
        "Bored": "#9B59B6"
    }

    emoji = {
        "Confused": "🤔",
        "Curious": "🧐",
        "Confident": "😎",
        "Frustrated": "😣",
        "Bored": "😴"
    }

    cols = st.columns(len(multilabel["mixed_emotions"]))

    for col, item in zip(cols, multilabel["mixed_emotions"]):

        emotion = item["emotion"]
        confidence = item["confidence"]

        color = colors.get(emotion, "#34495E")

        col.markdown(
            f"""
            <div style="
                background:{color};
                padding:20px;
                border-radius:15px;
                color:white;
                text-align:center;
                box-shadow:0px 3px 8px rgba(0,0,0,0.2);
            ">
                <h3>{emotion}</h3>
                <h2>{confidence:.1%}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    overall = " + ".join(
        [
            f"{emoji.get(x['emotion'], '🙂')} {x['emotion']}"
            for x in multilabel["mixed_emotions"]
        ]
    )

    st.success(f"### Overall Emotional State\n\n{overall}")

    # =====================================================
    # CONFIDENCE BARS
    # =====================================================

    if show_details:

        st.markdown("---")
        st.subheader("📊 Emotion Confidence Scores")

        sorted_scores = sorted(
            multilabel["scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )

        for emotion, score in sorted_scores:

            st.write(
                f"**{emotion.capitalize()}** — {score:.1%}"
            )

            st.progress(
                float(min(score, 1.0))
            )
# =====================================================
# GEMINI AI RESPONSE
# =====================================================

if analyze:

    if use_ai:

        st.markdown("---")
        st.subheader("🤖 AI Learning Guidance")

        detected_emotions = [
            x["emotion"]
            for x in multilabel["mixed_emotions"]
        ]

        with st.spinner("Generating AI Guidance..."):

            response = get_gemini_response(
                field,
                question,
                detected_emotions
            )

        st.info(response)

        # =====================================
        # Personalized Study Plan
        # =====================================

        st.markdown("---")
        st.subheader("📅 Personalized Study Plan")

        emotion = multilabel["emotion"].lower()

        if emotion == "confused":

            st.success("""
### Today's Study Plan

✅ Read the concept for 20 minutes

✅ Watch one beginner-friendly YouTube explanation

✅ Draw diagrams or write short notes

✅ Solve 5 Easy Questions

✅ Revise mistakes

Estimated Time: 1 Hour
""")

        elif emotion == "frustrated":

            st.success("""
### Today's Study Plan

✅ Take a 10-minute break

✅ Review the concept slowly

✅ Solve only Easy Problems

✅ Don't attempt Hard Problems today

✅ Practice again after a short break

Estimated Time: 45 Minutes
""")

        elif emotion == "bored":

            st.success("""
### Today's Study Plan

✅ Learn using videos

✅ Try interactive quizzes

✅ Build one small mini-project

✅ Solve one coding challenge

Estimated Time: 1 Hour
""")

        elif emotion == "curious":

            st.success("""
### Today's Study Plan

✅ Learn advanced concepts

✅ Read official documentation

✅ Build one mini-project

✅ Solve Medium Problems

Estimated Time: 1.5 Hours
""")

        elif emotion == "confident":

            st.success("""
### Today's Study Plan

✅ Solve Medium & Hard Problems

✅ Attempt Mock Interview Questions

✅ Build a Project

✅ Teach someone else

Estimated Time: 2 Hours
""")

        else:

            st.success("""
### Today's Study Plan

✅ Continue practicing

✅ Revise notes

✅ Solve coding problems

Estimated Time: 1 Hour
""")

        # =====================================
        # Download AI Report
        # =====================================

        report = f"""
AI LEARNING ASSISTANT REPORT
==================================================

Subject:
{field}

Student Question:
{question}

Primary Emotion:
{multilabel["emotion"]}

Confidence:
{multilabel["confidence"]:.2%}

Detected Emotions:
{", ".join([x["emotion"] for x in multilabel["mixed_emotions"]])}

==================================================

AI GUIDANCE

{response}

==================================================

Generated by:
AI Learning Assistant
BERT + Multi-label BiLSTM + Gemini AI
"""

        st.download_button(
            label="📄 Download AI Report",
            data=report,
            file_name="AI_Report.txt",
            mime="text/plain",
            use_container_width=True
        )

    else:

        response = ""         

# =====================================================
# SAVE HISTORY
# =====================================================

if analyze:

    if save_csv:

        save_history(
            field=field,
            text=question,
            emotion=multilabel["emotion"],
            confidence=multilabel["confidence"],
            model="BiLSTM Multi-label",
            response=response
        )

    st.session_state.emotion_history.append({
        "Field": field,
        "Question": question,
        "Primary Emotion": multilabel["emotion"],
        "Mixed Emotions": ", ".join(
            [x["emotion"] for x in multilabel["mixed_emotions"]]
        ),
        "Confidence": round(multilabel["confidence"] * 100, 2),
        "Model": "BiLSTM Multi-label"
    })

# =====================================================
# SESSION HISTORY
# =====================================================

st.markdown("---")
st.subheader("📜 Session History")

if len(st.session_state.emotion_history) > 0:

    history_df = pd.DataFrame(
        st.session_state.emotion_history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

else:

    st.info("No session history available.")

# =====================================================
# DOWNLOAD CSV
# =====================================================

if os.path.exists(HISTORY_FILE):

    with open(HISTORY_FILE, "rb") as file:

        st.download_button(
            "⬇ Download History CSV",
            data=file,
            file_name="history.csv",
            mime="text/csv",
            use_container_width=True
        )

# =====================================================
# =====================================================
# ANALYTICS DASHBOARD
# =====================================================

st.markdown("---")

tab1, tab2, tab3 = st.tabs([
    "😊 Emotion Analytics",
    "📚 Subject Analytics",
    "📊 Summary"
])

# =====================================================
# TAB 1 : Emotion Analytics
# =====================================================

with tab1:

    pie = emotion_pie_chart()

    if pie is not None:
        st.plotly_chart(
            pie,
            use_container_width=True,
            key="pie_chart"
        )

    bar = emotion_chart()

    if bar is not None:
        st.plotly_chart(
            bar,
            use_container_width=True,
            key="bar_chart"
        )

    timeline = confidence_timeline()

    if timeline is not None:
        st.plotly_chart(
            timeline,
            use_container_width=True,
            key="timeline_chart"
        )
# =====================================================
# TAB 2 : Subject Analytics
# =====================================================

with tab2:

    field_fig = field_chart()

    if field_fig is not None:
        st.plotly_chart(
            field_fig,
            use_container_width=True,
            key="field_chart"
        )
    else:
        st.info("No field analytics available.")
# =====================================================
# TAB 3 : Summary
# =====================================================

with tab3:

    stats = summary_stats()

    st.write("Summary Stats:", stats)

    if stats is not None:

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Total Sessions",
                stats.get("Total Sessions", 0)
            )

            st.metric(
                "Most Common Emotion",
                stats.get("Most Common Emotion", "-")
            )

        with c2:

            st.metric(
                "Most Studied Field",
                stats.get("Most Studied Field", "-")
            )

            st.metric(
                "Average Confidence",
                f"{stats.get('Average Confidence', 0)}%"
            )

    else:

        st.info("No statistics available.")
# =====================================================
# CURRENT SESSION SUMMARY
# =====================================================

if len(st.session_state.emotion_history) > 0:

    st.markdown("---")
    st.subheader("📈 Current Session")

    total = len(st.session_state.emotion_history)

    emotions = []

    for row in st.session_state.emotion_history:

        if "Primary Emotion" in row:
            emotions.append(row["Primary Emotion"])

        elif "emotion" in row:
            emotions.append(row["emotion"])

    # Find most common emotion
    if len(emotions) > 0:
        most_common = max(set(emotions), key=emotions.count)
    else:
        most_common = "No Data"

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Questions Asked",
            total
        )

    with c2:
        st.metric(
            "Most Frequent Emotion",
            most_common
        )

# =====================================================
# ABOUT PROJECT
# =====================================================

st.markdown("---")

with st.expander("ℹ About This Project"):

    st.markdown("""
# AI Learning Assistant

### Features

- 🎓 BERT Emotion Detection
- 🧠 Multi-label BiLSTM
- 😊 Mixed Emotion Detection
- 🤖 Google Gemini AI Guidance
- 📊 Interactive Analytics
- 📈 Emotion Dashboard
- 💾 CSV History
- 📜 Session History

### Technologies

- Python
- Streamlit
- TensorFlow
- Hugging Face Transformers
- Google Gemini
- Plotly
""")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown("""
<center>

## 🎓 AI Learning Assistant

Built with

🤖 BERT | 🧠 Multi-label BiLSTM | 💡 Gemini AI | 📊 Streamlit

</center>
""", unsafe_allow_html=True)