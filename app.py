import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Health Condition Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =================================================================
# STYLE — one CSS block controls the whole look. Streamlit's grid
# (st.columns) already reflows to a single column on narrow /
# mobile screens, so this stays responsive without extra media
# queries; the few we add just tighten spacing on small screens.
# =================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #eaf6ff 0%, transparent 60%),
                    radial-gradient(1000px 500px at 110% 10%, #f1ecff 0%, transparent 55%),
                    #f7f9fc;
    }

    #MainMenu, footer, header { visibility: hidden; }

    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 980px; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #4f7cff 0%, #7b5cff 55%, #b56bff 100%);
        border-radius: 24px;
        padding: 2.4rem 2.2rem;
        color: white;
        box-shadow: 0 20px 45px -18px rgba(79, 92, 255, 0.55);
        margin-bottom: 1.8rem;
    }
    .hero h1 { font-weight: 800; font-size: 2rem; margin: 0 0 0.4rem 0; }
    .hero p { font-weight: 400; font-size: 1.02rem; opacity: 0.92; margin: 0; max-width: 640px; }
    .hero-badges { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
    .hero-badge {
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Section cards */
    .section-card {
        background: white;
        border-radius: 18px;
        padding: 1.4rem 1.5rem 0.6rem 1.5rem;
        box-shadow: 0 8px 24px -12px rgba(30, 41, 59, 0.10);
        border: 1px solid #eef1f6;
        margin-bottom: 1.2rem;
    }
    .section-title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #1e2a4a;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #4f7cff, #8a5cff);
        color: white;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1.2rem;
        font-weight: 600;
        font-size: 1.02rem;
        box-shadow: 0 12px 24px -10px rgba(94, 92, 255, 0.55);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px -10px rgba(94, 92, 255, 0.65);
        color: white;
        border: none;
    }

    /* Result banner */
    .result-banner {
        border-radius: 20px;
        padding: 1.6rem 1.8rem;
        margin-top: 0.6rem;
        margin-bottom: 1.2rem;
        color: white;
        box-shadow: 0 16px 32px -16px rgba(0,0,0,0.35);
    }
    .result-banner h2 { margin: 0 0 0.3rem 0; font-weight: 800; font-size: 1.6rem; }
    .result-banner p { margin: 0; opacity: 0.95; }

    .fit-bg      { background: linear-gradient(135deg, #22c55e, #16a34a); }
    .risk-bg     { background: linear-gradient(135deg, #f59e0b, #ea580c); }
    .unhealthy-bg{ background: linear-gradient(135deg, #ef4444, #dc2626); }

    /* Sliders / labels */
    .stSlider label, .stNumberInput label, .stSelectbox label {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.92rem !important;
    }

    div[data-baseweb="select"] > div { border-radius: 10px !important; }

    .footnote { text-align: center; color: #94a3b8; font-size: 0.82rem; margin-top: 2rem; }

    @media (max-width: 640px) {
        .hero { padding: 1.6rem 1.4rem; border-radius: 18px; }
        .hero h1 { font-size: 1.5rem; }
        .section-card { padding: 1.1rem 1.1rem 0.3rem 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =================================================================
# LOAD MODEL
# =================================================================
@st.cache_resource
def load_artifact():
    return joblib.load("models/health_condition_model.pkl")

artifact = load_artifact()
model = artifact["model"]
feature_cols = artifact["feature_cols"]
categorical_cols = artifact["categorical_cols"]
target_map_inv = artifact["target_map_inv"]

# =================================================================
# HERO
# =================================================================
st.markdown(
    """
    <div class="hero">
        <h1>🩺 Health Condition Predictor</h1>
        <p>Enter a few lifestyle and health metrics and get an instant, AI-powered read on
        whether someone trends <b>fit</b>, <b>at-risk</b>, or <b>unhealthy</b> — trained on
        roughly 690,000 real health records.</p>
        <div class="hero-badges">
            <span class="hero-badge">⚡ LightGBM · Optuna-tuned</span>
            <span class="hero-badge">🎯 ~94.9% balanced accuracy</span>
            <span class="hero-badge">🔒 Runs locally, no data stored</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =================================================================
# INPUTS — grouped into clear cards, two responsive columns each
# =================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📏 Body & Vitals</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    bmi = st.slider("BMI", 15.0, 40.0, 23.0, 0.1)
with c2:
    heart_rate = st.slider("Resting heart rate (bpm)", 40.0, 150.0, 75.0, 1.0)
with c3:
    gender = st.selectbox("Gender", ["male", "female", "other"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🏃 Activity & Energy</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    step_count = st.number_input("Daily step count", 0, 30000, 8000, 100)
with c2:
    exercise_duration = st.slider("Exercise duration (min/day)", 0.0, 180.0, 40.0, 1.0)
with c3:
    physical_activity_level = st.selectbox("Activity level", ["sedentary", "moderate", "active"])
c4, c5 = st.columns(2)
with c4:
    calorie_expenditure = st.number_input("Calorie expenditure (kcal/day)", 500, 6000, 2200, 10)
with c5:
    water_intake = st.slider("Water intake (litres/day)", 0.0, 6.0, 2.2, 0.1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">😴 Sleep & Wellbeing</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    sleep_duration = st.slider("Sleep duration (hours)", 0.0, 12.0, 7.0, 0.1)
with c2:
    sleep_quality = st.selectbox("Sleep quality", ["poor", "average", "good"])
with c3:
    stress_level = st.selectbox("Stress level", ["low", "medium", "high"])
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🍽️ Diet & Habits</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    diet_type = st.selectbox("Diet type", ["veg", "non-veg", "balanced"])
with c2:
    smoking_alcohol = st.selectbox("Smoking / alcohol use", ["no", "occasional", "yes"])
st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("✨ Predict health condition", type="primary", use_container_width=True)


def engineer_features(raw: dict) -> pd.DataFrame:
    """Rebuild the exact same engineered features used in 02_feature_engineering.ipynb,
    so the live app matches the training-time feature pipeline exactly."""
    row = dict(raw)

    row["stress_activity_combo"] = f"{raw['stress_level']}_{raw['physical_activity_level']}"

    sd = raw["sleep_duration"]
    if sd <= 4:
        row["sleep_duration_bin"] = "very_low"
    elif sd <= 6:
        row["sleep_duration_bin"] = "low"
    elif sd <= 8:
        row["sleep_duration_bin"] = "good"
    else:
        row["sleep_duration_bin"] = "high"

    bmi_val = raw["bmi"]
    if bmi_val <= 18.5:
        row["bmi_category"] = "underweight"
    elif bmi_val <= 25:
        row["bmi_category"] = "normal"
    elif bmi_val <= 30:
        row["bmi_category"] = "overweight"
    else:
        row["bmi_category"] = "obese"

    row["activity_score"] = raw["step_count"] / 1000 + raw["exercise_duration"] / 10
    row["calorie_per_step"] = raw["calorie_expenditure"] / (raw["step_count"] + 1)

    df = pd.DataFrame([row])
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    return df[feature_cols]


# =================================================================
# RESULT
# =================================================================
if predict_clicked:
    raw_input = {
        "sleep_duration": sleep_duration,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "calorie_expenditure": calorie_expenditure,
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "water_intake": water_intake,
        "diet_type": diet_type,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "physical_activity_level": physical_activity_level,
        "smoking_alcohol": smoking_alcohol,
        "gender": gender,
    }

    X_input = engineer_features(raw_input)
    probs = model.predict_proba(X_input)[0]
    pred_class = probs.argmax()
    pred_label = target_map_inv[pred_class]
    confidence = probs[pred_class]

    style_map = {
        "fit": ("fit-bg", "🟢", "Fit",
                "Great signs across the board — keep the current routine going."),
        "at-risk": ("risk-bg", "🟡", "At-risk",
                    "Some indicators are drifting the wrong way — small, consistent changes now go a long way."),
        "unhealthy": ("unhealthy-bg", "🔴", "Unhealthy",
                      "Several indicators suggest real strain on the body — worth a conversation with a healthcare professional."),
    }
    css_class, emoji, title, blurb = style_map.get(pred_label, ("risk-bg", "🟡", pred_label, ""))

    st.markdown(
        f"""
        <div class="result-banner {css_class}">
            <h2>{emoji} {title} &nbsp;·&nbsp; {confidence*100:.1f}% confidence</h2>
            <p>{blurb}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    prob_df = pd.DataFrame({
        "Class": [target_map_inv[i].title() for i in range(len(probs))],
        "Probability": probs,
    }).sort_values("Probability", ascending=True)

    color_lookup = {"Fit": "#22c55e", "At-Risk": "#f59e0b", "Unhealthy": "#ef4444"}
    bar_colors = [color_lookup.get(c, "#6366f1") for c in prob_df["Class"]]

    fig = go.Figure(go.Bar(
        x=prob_df["Probability"],
        y=prob_df["Class"],
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in prob_df["Probability"]],
        textposition="outside",
        textfont=dict(color="#1e293b", size=13),
        cliponaxis=False,
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=10, r=50, t=10, b=30),
        xaxis=dict(
            range=[0, 1], tickformat=".0%", showgrid=False,
            tickfont=dict(color="#334155", size=12),
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color="#1e293b", size=13),
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, sans-serif", size=13, color="#1e293b"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "Model: LightGBM (Optuna-tuned), trained with 5-fold cross-validation, "
        "OOF balanced accuracy ≈ 0.949 on the training set."
    )

st.markdown(
    '<div class="footnote">This tool offers a statistical estimate, not a medical diagnosis. '
    'For any health concerns, please consult a qualified professional.</div>',
    unsafe_allow_html=True,
)