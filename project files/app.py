import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from src.database import init_db, get_db, User, Emotion_Record
from sqlalchemy.orm import Session

load_dotenv()

from src.preprocessing import get_mixed_emotions

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🎓",
    layout="wide"
)

def apply_custom_css():
    st.markdown("""
        <style>
        /* Modern Button Styling */
        div.stButton > button:first-child {
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(108, 99, 255, 0.2);
        }
        
        /* Metric Glassmorphism */
        div[data-testid="metric-container"] {
            background-color: rgba(38, 39, 48, 0.5);
            border: 1px solid rgba(250, 250, 250, 0.1);
            padding: 1rem;
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            border-right: 1px solid rgba(250, 250, 250, 0.1);
        }
        
        /* Header spacing */
        h1, h2, h3 {
            padding-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants (Section 5.2)
# ---------------------------------------------------------------------------

ACADEMIC_FIELDS = [
    "Computer Science", "Mathematics", "Physics", "Chemistry", "Biology",
    "Engineering", "Business", "Literature", "History", "Psychology", "Other"
]

EMOTION_RESPONSES = {
    'Confused': {
        'emoji': '😕',
        'response': 'I see you might be confused. Let me break this down step-by-step...',
        'action': 'Show detailed explanation'
    },
    'Frustrated': {
        'emoji': '😤',
        'response': 'I understand this is frustrating! Let\'s try a simpler approach...',
        'action': 'Suggest alternative learning path'
    },
    'Confident': {
        'emoji': '🎉',
        'response': 'Great! You\'re making excellent progress! Ready for the next challenge?',
        'action': 'Suggest advanced content'
    },
    'Bored': {
        'emoji': '😑',
        'response': 'Let\'s make this more engaging. Here are some interactive exercises...',
        'action': 'Show interactive content'
    },
    'Curious': {
        'emoji': '🤩',
        'response': 'Excellent question! Here\'s more in-depth information...',
        'action': 'Provide research papers & advanced materials'
    }
}

CSV_EXAMPLES_FILE = 'emotion_response_examples.csv'
CSV_MAPPING_FILE = 'emotion_response_mapping.csv'

# ---------------------------------------------------------------------------
# Model Loading — lazy import so missing models show friendly error (Section 9)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    """Load both models once and cache them across all Streamlit sessions."""
    try:
        import torch
        torch.set_num_threads(1)
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        from src.model import load_bilstm_model
        from src.bert_model import load_bert_model
        bilstm_assets = load_bilstm_model()
        bert_assets = load_bert_model()
        return bilstm_assets, bert_assets
    except FileNotFoundError as e:
        st.error(
            "⚠️ **Model files not found.**\n\n"
            "Please complete Kaggle training first and place the model files in:\n"
            "- `models/bltsm/` — BiLSTM model, tokenizer, label_classes\n"
            "- `models/bert_emotion_model_final/` — BERT model + tokenizer + label_mapping.json\n\n"
            f"Details: `{e}`"
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading models: `{e}`")
        st.stop()

# ---------------------------------------------------------------------------
# Gemini Integration (Section 5.4)
# ---------------------------------------------------------------------------

def setup_gemini():
    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY', '')
    if api_key:
        return genai.Client(api_key=api_key)
    return None


def get_gemini_response(model_gemini, field: str, problem: str, emotion: str, confidence: float) -> str:
    """Get AI response from Gemini based on field and problem."""
    if model_gemini is None:
        return EMOTION_RESPONSES[emotion]['response']
    try:
        prompt = f"""
        You are a helpful learning assistant. A student studying {field} is feeling {emotion} (confidence: {confidence:.1%}) about this problem:
        
        "{problem}"
        
        Provide a clear, supportive response with:
        1. Brief acknowledgment of their feeling
        2. One specific tip or strategy for {field}
        3. One encouraging next step
        
        Use simple, clear language. Keep each point to 1-2 sentences. No markdown formatting.
        """
        response = model_gemini.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"AI response unavailable: {e}"

# ---------------------------------------------------------------------------
# CSV Logging (Section 5.5)
# ---------------------------------------------------------------------------

def log_interaction(entry: dict, save_csv: bool):
    """Save to database and optionally to CSV."""
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        new_record = Emotion_Record(
            UserID=st.session_state.user_id,
            Email=st.session_state.user_email,
            Field=entry['field'],
            Input_Text=entry['input_text'],
            Predicted_Emotion=entry['emotion'],
            Confidence_Score=entry['confidence'],
            Model_Used=entry['model_used'],
            AI_Response=entry['ai_response'],
            Emotion_Scores=entry['emotion_scores'],
            Timestamp=datetime.datetime.fromisoformat(entry['timestamp']),
            CSV_Logged=save_csv
        )
        db.add(new_record)
        db.commit()
    except Exception as e:
        st.error(f"Database error: {e}")
    finally:
        db.close()

    if save_csv:
        new_example = {
            'text': entry['input_text'],
            'emotion': entry['emotion'].lower(),
            'confidence': entry['confidence'],
            'response': entry['ai_response'],
            'field': entry['field'],
            'timestamp': entry['timestamp']
        }
        
        if os.path.exists(CSV_EXAMPLES_FILE):
            df = pd.read_csv(CSV_EXAMPLES_FILE)
            df = pd.concat([df, pd.DataFrame([new_example])], ignore_index=True)
        else:
            df = pd.DataFrame([new_example])
            
        df.to_csv(CSV_EXAMPLES_FILE, index=False)
        
        # Update mapping CSV if new emotion-response pair
        if os.path.exists(CSV_MAPPING_FILE):
            mapping_df = pd.read_csv(CSV_MAPPING_FILE)
            if entry['emotion'] not in mapping_df['emotion'].values:
                new_mapping = pd.DataFrame([{'emotion': entry['emotion'], 'response': entry['ai_response']}])
                mapping_df = pd.concat([mapping_df, new_mapping], ignore_index=True)
                mapping_df.to_csv(CSV_MAPPING_FILE, index=False)
        else:
            mapping_df = pd.DataFrame([{'emotion': entry['emotion'], 'response': entry['ai_response']}])
            mapping_df.to_csv(CSV_MAPPING_FILE, index=False)

# ---------------------------------------------------------------------------
# Session State (Section 5.6)
# ---------------------------------------------------------------------------

def init_session_state():
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None

# ---------------------------------------------------------------------------
# Sidebar (Section 5.7)
# ---------------------------------------------------------------------------

def render_sidebar(gemini_model):
    with st.sidebar:
        st.title("🧠 AI Learning Assistant")
        st.markdown("---")

        if st.session_state.user_id is None:
            st.subheader("Login / Sign Up")
            auth_mode = st.radio("Select mode", ["Login", "Sign Up"], horizontal=True)
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            if auth_mode == "Sign Up":
                name = st.text_input("Name")
                if st.button("Sign Up"):
                    db = next(get_db())
                    existing = db.query(User).filter(User.Email == email).first()
                    if existing:
                        st.error("Email already registered.")
                    elif not email or not password or not name:
                        st.error("Please fill all fields.")
                    else:
                        new_user = User(Name=name, Email=email)
                        new_user.set_password(password)
                        db.add(new_user)
                        db.commit()
                        st.success("Account created! You can now log in.")
                    db.close()
            else:
                if st.button("Login"):
                    db = next(get_db())
                    user = db.query(User).filter(User.Email == email).first()
                    if user and user.check_password(password):
                        st.session_state.user_id = user.UserID
                        st.session_state.user_name = user.Name
                        st.session_state.user_email = user.Email
                        user.Login_Count += 1
                        db.commit()
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                    db.close()
            return False
        
        else:
            st.subheader(f"Welcome, {st.session_state.user_name} 👋")
            if st.button("Logout"):
                st.session_state.user_id = None
                st.session_state.user_name = None
                st.session_state.user_email = None
                st.rerun()

        st.markdown("---")

        # Model status
        st.header("📊 Dashboard")
        status = "Active" if gemini_model else "Template Fallback"
        st.write(f"Models: {status}")

        csv_count = 0
        if os.path.exists(CSV_EXAMPLES_FILE):
            try:
                csv_count = len(pd.read_csv(CSV_EXAMPLES_FILE))
            except Exception:
                csv_count = 0

        st.write(f"Total Interactions: {len(st.session_state.emotion_history)}")
        st.write(f"CSV Examples: {csv_count}")

        if st.button("Clear History"):
            st.session_state.emotion_history = []
            st.rerun()

        # Recent interactions
        if st.session_state.emotion_history:
            st.subheader("Recent Sessions")
            recent = st.session_state.emotion_history[-3:]
            for item in reversed(recent):
                st.write(f"• {item['field']}: {item['emotion']} ({item['confidence']:.1%})")



# ---------------------------------------------------------------------------
# Results Display (Section 5.10)
# ---------------------------------------------------------------------------

def add_to_history(field, problem, emotion, confidence, ai_response, bilstm_scores, bert_result=None):
    # Detect mixed emotions for history
    def get_mixed_emotions(scores, threshold=0.15):
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_emotions[0]
        mixed = [primary]
        
        for emotion_name, score in sorted_emotions[1:]:
            if score >= threshold:
                mixed.append((emotion_name, score))
                
        return mixed if len(mixed) > 1 else [primary]
        
    mixed_emotions = get_mixed_emotions(bilstm_scores)
    emotion_label = " + ".join([em[0] for em in mixed_emotions]) if len(mixed_emotions) > 1 else emotion
    
    # Add BiLSTM entry
    st.session_state.emotion_history.append({
        'timestamp': datetime.datetime.now(),
        'field': field,
        'problem': problem,
        'emotion': emotion_label,
        'confidence': confidence,
        'ai_response': ai_response,
        'all_scores': bilstm_scores,
        'model': 'BiLSTM'
    })
    
    # Add BERT entry if available
    if bert_result:
        bert_mixed = get_mixed_emotions(bert_result['scores'])
        bert_emotion_label = " + ".join([em[0] for em in bert_mixed]) if len(bert_mixed) > 1 else bert_result['emotion']
        
        st.session_state.emotion_history.append({
            'timestamp': datetime.datetime.now(),
            'field': field,
            'problem': problem,
            'emotion': bert_emotion_label,
            'confidence': bert_result['confidence'],
            'ai_response': ai_response,
            'all_scores': bert_result['scores'],
            'model': 'BERT'
        })


def render_results(prediction: dict, field: str, problem_text: str,
                   use_ai: bool, save_data: bool, show_details: bool,
                   gemini_model, override_ai_response: str = None):
    """
    Display BiLSTM vs BERT comparison, mixed emotions, AI response.
    Log to session history and CSV.
    """
    bilstm_result = prediction['bilstm_result']
    bert_result = prediction['bert_result']
    bilstm_mixed = prediction['bilstm_mixed']
    bert_mixed = prediction['bert_mixed']

    st.markdown("---")
    st.header("🔍 Emotion Analysis Results")

    # Model comparison with mixed sentiment detection
    st.subheader("🔬 Model Predictions Comparison")
    
    with st.container(border=True):
        if bert_result:
            col1, col2 = st.columns(2)
        else:
            col1 = st.columns(1)[0]
            
        with col1:
            st.write("**BiLSTM Student Adaptive**")
            bilstm_mixed = get_mixed_emotions(bilstm_result['scores'])
            
            if len(bilstm_mixed) > 1:
                mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bilstm_mixed])
                st.metric("Mixed Emotions", mixed_text, f"Primary: {bilstm_mixed[0][1]:.1%}")
            else:
                bilstm_emoji = EMOTION_RESPONSES[bilstm_result['emotion']]['emoji']
                st.metric("Emotion", f"{bilstm_emoji} {bilstm_result['emotion']}", f"{bilstm_result['confidence']:.1%}")
                
            for emotion_name, score in sorted(bilstm_result['scores'].items(), key=lambda x: x[1], reverse=True):
                st.progress(score, text=f"{emotion_name}: {score:.1%}")
    
        if bert_result:
            with col2:
                st.write("**BERT Transformer**")
                bert_mixed = get_mixed_emotions(bert_result['scores'])
                
                if len(bert_mixed) > 1:
                    mixed_text = " + ".join([f"{EMOTION_RESPONSES[em[0]]['emoji']} {em[0]}" for em in bert_mixed])
                    st.metric("Mixed Emotions", mixed_text, f"Primary: {bert_mixed[0][1]:.1%}")
                else:
                    bert_emoji = EMOTION_RESPONSES[bert_result['emotion']]['emoji']
                    st.metric("Emotion", f"{bert_emoji} {bert_result['emotion']}", f"{bert_result['confidence']:.1%}")
                    
                for emotion_name, score in sorted(bert_result['scores'].items(), key=lambda x: x[1], reverse=True):
                    st.progress(score, text=f"{emotion_name}: {score:.1%}")

    # AI Response
    st.markdown("---")
    st.subheader("💬 Personalized Learning Support")

    primary_emotion = bilstm_result['emotion']
    primary_confidence = bilstm_result['confidence']

    if override_ai_response:
        ai_response = override_ai_response
    elif use_ai:
        with st.spinner("Generating personalized guidance..."):
            ai_response = get_gemini_response(
                gemini_model, field, problem_text, primary_emotion, primary_confidence
            )
    else:
        ai_response = EMOTION_RESPONSES[primary_emotion]['response']

    emoji = EMOTION_RESPONSES[primary_emotion]['emoji']
    
    with st.chat_message("assistant", avatar=emoji):
        st.markdown(f"**AI Response based on BiLSTM prediction: {primary_emotion}**")
        st.write(ai_response)
        
    st.success(f"📖 **Additional Support**\n\n**Strategy:** {EMOTION_RESPONSES[primary_emotion]['action']}")

    if show_details:
        with st.expander("🔬 Analysis Details"):
            st.json({
                'bilstm': bilstm_result,
                'bert': bert_result
            })

    # Log to session and CSV
    timestamp = datetime.datetime.now().isoformat()

    add_to_history(field, problem_text, primary_emotion, primary_confidence, ai_response, bilstm_result['scores'], bert_result)

    for model_label, result in [("BiLSTM", bilstm_result), ("BERT", bert_result)]:
        entry = {
            'timestamp': timestamp,
            'field': field,
            'input_text': problem_text,
            'emotion': result['emotion'],
            'confidence': result['confidence'],
            'model_used': model_label,
            'ai_response': ai_response,
            'emotion_scores': json.dumps(result['scores'])
        }
        log_interaction(entry, save_data)

# ---------------------------------------------------------------------------
# Analytics Dashboard (Section 5.11)
# ---------------------------------------------------------------------------

def render_analytics():
    """Display analytics dashboard with Plotly charts. Only shows when history exists."""
    if not st.session_state.emotion_history:
        return

    st.markdown("---")
    st.header("📈 Learning Analytics")

    df = pd.DataFrame(st.session_state.emotion_history)
    
    # Filter for primary model to avoid double counting
    if 'model' in df.columns:
        df_primary = df[df['model'] == 'BiLSTM']
    else:
        df_primary = df

    tab1, tab2, tab3 = st.tabs(["My Journey", "Subject Analysis", "Summary"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Emotion distribution pie chart
            emotion_counts = df_primary['emotion'].value_counts()
            fig1 = px.pie(
                values=emotion_counts.values, names=emotion_counts.index,
                title="Overall Mood Distribution",
                hole=0.4
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Emotion Timeline Scatter Plot
            df_copy = df_primary.copy()
            df_copy['time'] = pd.to_datetime(df_copy['timestamp']).dt.strftime('%H:%M:%S')
            fig2 = px.scatter(
                df_copy, x='time', y='emotion', color='emotion',
                title="Emotional Journey Timeline",
                size_max=10
            )
            fig2.update_traces(marker=dict(size=12))
            fig2.update_yaxes(categoryorder="total ascending")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        # Emotions by Study Field (Stacked Bar)
        field_emotion = df_primary.groupby(['field', 'emotion']).size().reset_index(name='count')
        fig3 = px.bar(
            field_emotion, x='field', y='count', color='emotion',
            title="Emotions by Subject",
            barmode="stack"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interactions", len(df_primary))
        with col2:
            avg_conf = df_primary['confidence'].mean()
            st.metric("Avg Prediction Confidence", f"{avg_conf:.1%}")
        with col3:
            top_emotion = df_primary['emotion'].mode()[0] if len(df_primary) > 0 else "—"
            st.metric("Most Frequent Emotion", top_emotion)
            
        # Actionable insight
        if len(df_primary) > 0:
            st.markdown("### Actionable Insight")
            if top_emotion == "Frustrated":
                st.info("💡 **Insight**: You've been feeling mostly Frustrated. Consider taking a 5-minute break, reviewing fundamental concepts, or trying the Pomodoro technique to avoid burnout!")
            elif top_emotion == "Confused":
                st.info("💡 **Insight**: You seem Confused frequently. Try breaking down problems into smaller steps or exploring a different explanation (like a YouTube video) for these topics.")
            elif top_emotion == "Bored":
                st.info("💡 **Insight**: You're feeling Bored. Try to challenge yourself with harder problems or find a way to apply what you're learning to a real-world project that interests you.")
            else:
                st.success(f"💡 **Insight**: Your dominant emotion is {top_emotion}. Keep up the great work and maintain your current study strategies!")

        st.dataframe(
            df_primary[['timestamp', 'field', 'emotion', 'confidence', 'model']].tail(10),
            use_container_width=True
        )

# ---------------------------------------------------------------------------
# Main (Section 5.12)
# ---------------------------------------------------------------------------

def main():
    # Initialize Database
    init_db()

    init_session_state()

    # Load models (cached)
    with st.spinner("Loading models..."):
        bilstm_assets, bert_assets = load_models()

    # Setup Gemini
    gemini_model = setup_gemini()

    # Render sidebar
    auth_status = render_sidebar(gemini_model)

    # Page title
    st.title("🤖 Emotion-Aware Learning Assistant")
    
    apply_custom_css()

    if auth_status is False:
        st.info("👋 Please login or sign up from the sidebar to continue.")
        return

    st.markdown("Get personalized help based on your field and emotional state")

    # Load examples globally for sidebar & settings usage
    examples_df = []
    if os.path.exists(CSV_EXAMPLES_FILE):
        try:
            examples_df = pd.read_csv(CSV_EXAMPLES_FILE)
        except Exception:
            pass

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            st.subheader("📚 Tell us about your learning challenge")
            # Field selection
            field = st.selectbox(
                "What field are you studying?",
                ACADEMIC_FIELDS,
                help="Select your area of study for personalized responses"
            )
            
            # Problem description
            problem_text = st.text_area(
                f"Describe your {field} problem or challenge:",
                placeholder=f"e.g., 'I'm struggling with algorithms in {field}' or 'This concept is confusing'",
                height=120,
                key="problem_input"
            )
            
            st.write("**Quick Examples:**")
            ex1, ex2, ex3 = st.columns(3)
            
            def set_example(text):
                st.session_state.problem_input = text
                
            with ex1:
                st.button("I'm confused about recursion", on_click=set_example, args=("I'm confused about recursion",), use_container_width=True)
            with ex2:
                st.button("Debugging is frustrating", on_click=set_example, args=("Debugging is frustrating",), use_container_width=True)
            with ex3:
                st.button("I'm curious about machine learning", on_click=set_example, args=("I'm curious about machine learning",), use_container_width=True)

    with col2:
        with st.container(border=True):
            st.subheader("⚙️ Settings")
            use_ai = st.checkbox("Use AI Response (Gemini)", value=True)
            save_data = st.checkbox("Save to CSV for learning", value=True)
            show_details = st.checkbox("Show analysis details", value=False)
            
            # CSV Prediction Option
            st.markdown("---")
            st.write("**📊 Predict from Saved Data**")
            use_csv_prediction = st.checkbox("Use CSV-based prediction", value=False)
            
            if use_csv_prediction and len(examples_df) > 0:
                st.info(f"Using {len(examples_df)} saved examples for prediction")

    # Run button
    col1, col2 = st.columns([1, 5])
    with col1:
        run_clicked = st.button("🔍 Get AI Learning Help", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🔄 Clear", use_container_width=False)

    if clear_clicked:
        st.rerun()

    if run_clicked:
        # Edge case: empty or too-short input (Section 9)
        if not problem_text.strip() or len(problem_text.strip()) < 3:
            st.warning("Please describe your study challenge in at least a few words.")
        else:
            from src.predict import run_prediction
            
            prediction = None
            csv_ai_response = None
            
            if use_csv_prediction and len(examples_df) > 0:
                match = examples_df[examples_df['text'].str.lower() == problem_text.lower().strip()]
                if not match.empty:
                    match_row = match.iloc[-1]
                    csv_emotion = str(match_row['emotion']).capitalize()
                    csv_conf = float(match_row['confidence'])
                    prediction = {
                        'bilstm_result': {
                            'emotion': csv_emotion,
                            'confidence': csv_conf,
                            'scores': {csv_emotion: csv_conf},
                            'cleaned_text': problem_text
                        },
                        'bert_result': None,
                        'bilstm_mixed': [(csv_emotion, csv_conf)],
                        'bert_mixed': []
                    }
                    csv_ai_response = str(match_row['response'])
                    st.success("🎯 Exact match found in saved CSV dataset!")
                    
            if prediction is None:
                with st.spinner("Analyzing your emotions..."):
                    prediction = run_prediction(problem_text, bilstm_assets, bert_assets)
                    
            render_results(
                prediction, field, problem_text,
                use_ai, save_data, show_details,
                gemini_model, override_ai_response=csv_ai_response
            )

    # Analytics
    render_analytics()


if __name__ == "__main__":
    main()
