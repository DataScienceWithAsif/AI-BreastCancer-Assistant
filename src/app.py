import streamlit as st
from predictive_model import PredictiveModel
from assistant import BreastCancerAssistant, explain_prediction

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Breast Health Companion",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL STYLING
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: radial-gradient(circle at 10% 0%, #201a2b 0%, #14101c 45%, #0d0a13 100%);
    color: #f1eef7;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1526 0%, #120d18 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

.hero {
    padding: 2.2rem 2rem 1.6rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(214,51,132,0.18) 0%, rgba(124,58,237,0.18) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.8rem;
}
.hero h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 600;
    font-size: 2.1rem;
    margin-bottom: 0.3rem;
    background: linear-gradient(90deg, #f472b6, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #cbc4d9;
    font-size: 1rem;
    margin: 0;
}

.glass-card {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(6px);
}

.section-title {
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    font-size: 1.15rem;
    color: #eae6f5;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

div[data-testid="stSlider"] label {
    color: #cfc9e0 !important;
    font-size: 0.85rem !important;
    font-weight: 500;
}

.stButton > button {
    background: linear-gradient(90deg, #ec4899, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 1.6rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 18px rgba(236,72,153,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(139,92,246,0.35);
}

.result-card {
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-top: 1.2rem;
    border: 1px solid rgba(255,255,255,0.1);
    animation: fadeIn 0.4s ease;
}
.result-benign {
    background: linear-gradient(135deg, rgba(34,197,94,0.16), rgba(16,185,129,0.10));
    border-color: rgba(34,197,94,0.35);
}
.result-malignant {
    background: linear-gradient(135deg, rgba(239,68,68,0.16), rgba(220,38,38,0.10));
    border-color: rgba(239,68,68,0.35);
}
.result-label {
    font-family: 'Poppins', sans-serif;
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.result-confidence {
    color: #cfc9e0;
    font-size: 0.95rem;
}
.disclaimer {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.85rem;
    color: #fcd34d;
    margin-top: 1.2rem;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.chat-disclaimer {
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #ddd6fe;
    margin-bottom: 1rem;
}

/* --- Fix: chat input text was invisible (same color as background) --- */
[data-testid="stChatInput"] {
    background: #1c1526 !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: #1c1526 !important;
    color: #f1eef7 !important;
    caret-color: #f1eef7 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8b83a3 !important;
}

/* --- Fix: add vertical spacing between sidebar nav options --- */
div[data-testid="stRadio"] > div {
    gap: 0.6rem;
}
div[data-testid="stRadio"] label {
    padding: 0.55rem 0.8rem;
    border-radius: 10px;
    margin-bottom: 0.3rem;
    transition: background 0.15s ease;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,0.05);
}

/* --- Fix: chat bubbles blended into background, add contrast --- */
[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.7rem;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Assistant bubble: violet-tinted card, clearly offset from page background */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background: linear-gradient(135deg, rgba(139,92,246,0.14), rgba(236,72,153,0.08));
    border-color: rgba(167,139,250,0.25);
}

/* User bubble: neutral lighter card, distinct from assistant */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(255,255,255,0.06);
    border-color: rgba(255,255,255,0.12);
}

[data-testid="stChatMessage"] p {
    color: #f1eef7 !important;
    font-size: 0.98rem;
    line-height: 1.55;
}

/* Fallback for Streamlit versions where the avatar testid selector above doesn't match:
   alternate bubble color by position (user msgs are always odd, assistant always even,
   since they're appended in strict user->assistant pairs) */
[data-testid="stChatMessage"]:nth-of-type(odd) {
    background: rgba(255,255,255,0.06);
}
[data-testid="stChatMessage"]:nth-of-type(even) {
    background: linear-gradient(135deg, rgba(139,92,246,0.14), rgba(236,72,153,0.08));
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FEATURE RANGES  (min, max, default = midpoint)
# ----------------------------------------------------------------------------
FEATURE_RANGES = {
    "worst concave points":     (0.000000, 0.291000),
    "worst area":                (185.200000, 4254.000000),
    "mean perimeter":            (43.790000, 188.500000),
    "mean texture":              (9.710000, 39.280000),
    "area error":                (6.802000, 542.200000),
    "texture error":             (0.360200, 4.885000),
    "perimeter error":           (0.757000, 21.980000),
    "worst concavity":           (0.000000, 1.252000),
    "worst texture":             (12.020000, 49.540000),
    "worst perimeter":           (50.410000, 251.200000),
    "worst compactness":         (0.027290, 1.058000),
    "radius error":              (0.111500, 2.873000),
    "mean fractal dimension":    (0.049960, 0.097440),
    "worst smoothness":          (0.071170, 0.222600),
    "mean concave points":       (0.000000, 0.201200),
}

# ----------------------------------------------------------------------------
# CACHED RESOURCE LOADERS  (load once, reuse across reruns)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading prediction model...")
def get_predictor():
    return PredictiveModel()

@st.cache_resource(show_spinner="Loading AI assistant (this can take a minute on first load)...")
def get_assistant():
    return BreastCancerAssistant()


# ----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎗️ Breast Health AI")
    st.markdown("<p style='color:#a99fc2; font-size:0.85rem;'>DevSphere AI Internship Capstone</p>", unsafe_allow_html=True)
    st.markdown("---")
    mode = st.radio(
        "Choose a service",
        ["🔬 Cancer Prediction", "💬 AI Assistant"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<p style='color:#8b83a3; font-size:0.78rem; line-height:1.5;'>"
        "This tool is for educational purposes only and does not replace "
        "professional medical diagnosis or advice."
        "</p>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# PAGE: PREDICTION
# ----------------------------------------------------------------------------
if mode == "🔬 Cancer Prediction":
    st.markdown("""
    <div class="hero">
        <h1>Breast Tissue Prediction</h1>
        <p>Enter cell nuclei measurements from a diagnostic report to get a model-based prediction.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Input Measurements</div>', unsafe_allow_html=True)
    st.caption("Adjust each slider to match the values from a diagnostic report. Ranges reflect the training data's observed min/max.")

    feature_names = list(FEATURE_RANGES.keys())
    input_values = {}
    cols = st.columns(3)

    for i, fname in enumerate(feature_names):
        fmin, fmax = FEATURE_RANGES[fname]
        default = round((fmin + fmax) / 2, 5)
        with cols[i % 3]:
            input_values[fname] = st.slider(
                fname.replace("_", " ").title(),
                min_value=float(fmin),
                max_value=float(fmax),
                value=float(default),
                format="%.5f",
                key=f"slider_{fname}",
            )

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Run Prediction", use_container_width=False)

    if predict_clicked:
        predictor = get_predictor()
        st.session_state.prediction_result = predictor.predict_label(input_values)
        st.session_state.prediction_explanation = None  # reset any old explanation

    if "prediction_result" in st.session_state and st.session_state.prediction_result is not None:
        result = st.session_state.prediction_result

        css_class = "result-benign" if result["result"] == "Benign" else "result-malignant"
        icon = "✅" if result["result"] == "Benign" else "⚠️"

        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="result-label">{icon} {result['result']}</div>
            <div class="result-confidence">Model confidence: {result['confidence']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(result["confidence"])

        st.markdown("""
        <div class="disclaimer">
        ⚠️ This prediction is generated by a machine learning model for educational purposes only.
        It is <b>not a medical diagnosis</b>. Please consult a qualified doctor for any real health concerns.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💬 Want this explained in plain language? Ask the AI Assistant"):
            if st.button("Explain this result with AI Assistant"):
                assistant = get_assistant()
                with st.spinner("Thinking..."):
                    st.session_state.prediction_explanation = explain_prediction(result, assistant)

            if st.session_state.get("prediction_explanation"):
                st.markdown(f"<div class='glass-card'>{st.session_state.prediction_explanation}</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE: AI ASSISTANT (CHATBOT)
# ----------------------------------------------------------------------------
else:
    st.markdown("""
    <div class="hero">
        <h1>AI Breast Health Assistant</h1>
        <p>Ask questions about breast cancer, screening, and treatment in a supportive, judgment-free space.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="chat-disclaimer">
    💜 This assistant provides general educational information only and cannot diagnose or give
    personalized medical advice. For anything specific to your health, please talk to your doctor.
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # render existing conversation
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🎗️" if msg["role"] == "assistant" else "🧑"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about breast health, screening, treatment...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        assistant = get_assistant()
        with st.chat_message("assistant", avatar="🎗️"):
            with st.spinner("Thinking..."):
                # pass prior turns (excluding this latest user message, added separately in .ask)
                history = st.session_state.chat_history[:-1]
                response = assistant.ask(user_input, history=history)
            st.markdown(response)

        st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()