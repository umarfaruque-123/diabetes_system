import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="HealthPredict AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD MODEL
# =====================================

@st.cache_resource
def load_files():
    model = joblib.load("diabetes_model.pkl")
    imputer = joblib.load("diabetes_imputer.pkl")
    features = joblib.load("feature_columns.pkl")

    features = [
        f for f in features if f != "Diabetes_Yes"
    ]

    return model, imputer, features

try:
    model, imputer, features = load_files()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# =====================================
# PREMIUM UI CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background: #0b1020;
    color: #f1f5f9;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: rgba(11,16,32,0.95);
}

.stApp p, .stApp label {
    color: #cbd5e1;
}

.stApp h1, .stApp h2, .stApp h3 {
    color: #ffffff !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#151d38,#101426);
    border-right: 1px solid #293354;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* MAIN HEADER */

.hero {
    background: linear-gradient(120deg,#202c53,#32245c,#173c55);
    border: 1px solid #414b7a;
    border-radius: 22px;
    padding: 32px;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(0,0,0,.25);
}

.hero h1 {
    color: white !important;
    font-size: 36px;
    font-weight: 850;
    margin: 0;
}

.hero p {
    color: #c7d2fe;
    margin-top: 10px;
    font-size: 15px;
}

/* STAT CARDS */

.stat-card {
    background: linear-gradient(145deg,#1c2745,#19213a);
    border: 1px solid #35436a;
    border-radius: 18px;
    padding: 23px;
    min-height: 140px;
    transition: .25s ease;
}

.stat-card:hover {
    border-color: #8b7cf6;
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(112,92,231,.15);
}

.stat-label {
    color: #a5b4fc;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.2px;
}

.stat-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 800;
    margin-top: 13px;
}

.stat-note {
    color: #94a3b8;
    font-size: 12px;
    margin-top: 5px;
}

/* SECTION CARDS */

.panel {
    background: #151e35;
    border: 1px solid #2e3b5b;
    border-radius: 18px;
    padding: 23px;
    margin-bottom: 18px;
}

/* INPUTS */

.stNumberInput input,
.stTextInput input {
    background: #202b48 !important;
    color: white !important;
    border: 1px solid #465477 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] > div {
    background: #202b48 !important;
    color: white !important;
    border-color: #465477 !important;
    border-radius: 10px !important;
}

.stSelectbox div[data-baseweb="select"] span {
    color: white !important;
}

[data-testid="stWidgetLabel"] p {
    color: #dbeafe !important;
    font-weight: 600;
}

/* FORM */

[data-testid="stForm"] {
    background: #151e35;
    border: 1px solid #344365;
    border-radius: 18px;
    padding: 24px;
}

/* BUTTONS */

.stButton button,
.stFormSubmitButton button {
    background: linear-gradient(100deg,#6855e7,#8b5cf6) !important;
    color: white !important;
    border: 1px solid #927df9 !important;
    border-radius: 11px !important;
    min-height: 45px;
    font-weight: 750;
    transition: .2s ease;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background: linear-gradient(100deg,#806ef4,#a78bfa) !important;
    box-shadow: 0 5px 18px rgba(139,92,246,.3);
}

/* TABS */

.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background: #151e35;
    border-radius: 12px;
    padding: 8px;
}

.stTabs [data-baseweb="tab"] {
    background: #202b48;
    border-radius: 9px;
    color: #cbd5e1;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background: #6855e7 !important;
    color: white !important;
}

/* RESULT */

.result-box {
    background: linear-gradient(135deg,#1b2948,#282451);
    border: 1px solid #6257a6;
    border-radius: 18px;
    padding: 28px;
    text-align: center;
}

.result-heading {
    color: #c4b5fd;
    font-size: 13px;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.result-value {
    color: white;
    font-size: 29px;
    font-weight: 850;
    margin-top: 12px;
}

/* FOOTER */

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 12px;
}

hr {
    border-color: #303b59 !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR NAVIGATION
# =====================================

with st.sidebar:

    st.markdown("# 🧬 HealthPredict")

    st.caption("INTELLIGENT HEALTHCARE AI")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🩺 Prediction",
            "📘 About Project"
        ],
        label_visibility="visible"
    )

    st.divider()

    st.markdown("### MODEL STATUS")

    st.success("● Model Online")

    st.markdown("### ALGORITHM")

    st.write(type(model).__name__)

    st.markdown("### FEATURES")

    st.write(f"{len(features)} input features")

    st.divider()

    st.caption("MCA | Machine Learning Project")

# =====================================
# DASHBOARD PAGE
# =====================================

if page == "🏠 Dashboard":

    st.markdown("""
    <div class="hero">
        <h1>🧬 HealthPredict AI</h1>
        <p>
        Next Generation Diabetes Risk Assessment Platform.
        Intelligent predictions powered by Machine Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">SYSTEM STATUS</div>
            <div class="stat-value">● ACTIVE</div>
            <div class="stat-note">Model loaded successfully</div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">AI ALGORITHM</div>
            <div class="stat-value">Random Forest</div>
            <div class="stat-note">Classification model</div>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">INPUT PARAMETERS</div>
            <div class="stat-value">{len(features)}</div>
            <div class="stat-note">Patient health features</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("## ✨ Welcome to HealthPredict")

    st.markdown("""
    <div class="panel">
        <h3>About the Dashboard</h3>
        <p>
        HealthPredict uses a trained machine learning model
        to estimate the likelihood of a positive diabetes
        classification from patient health information.
        </p>
        <p>
        Enter patient details in the Prediction section
        to generate a model result.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Start Patient Assessment →"):
        st.session_state["go_prediction"] = True
        st.rerun()

    if st.session_state.get("go_prediction", False):
        st.info("Select 🩺 Prediction from the sidebar.")

# =====================================
# PREDICTION PAGE
# =====================================

elif page == "🩺 Prediction":

    st.markdown("""
    <div class="hero">
        <h1>🩺 Patient Assessment</h1>
        <p>
        Complete the health profile to generate
        a machine learning prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "👤 Personal Details",
        "🧪 Clinical Details",
        "🏃 Lifestyle Details"
    ])

    with st.form("patient_form"):

        with tab1:

            st.markdown("### Personal Information")

            col1, col2, col3 = st.columns(3)

            with col1:
                age = st.number_input(
                    "Age",
                    1, 120, 30
                )

            with col2:
                gender = st.selectbox(
                    "Gender",
                    ["Female", "Male", "Unknown"]
                )

            with col3:
                family = st.selectbox(
                    "Family History",
                    ["No", "Yes"]
                )

        with tab2:

            st.markdown("### Health Measurements")

            col1, col2 = st.columns(2)

            with col1:
                bmi = st.number_input(
                    "BMI",
                    10.0, 70.0, 25.0,
                    step=0.1
                )

                glucose = st.number_input(
                    "Glucose",
                    40.0, 400.0, 100.0
                )

                bp = st.number_input(
                    "Blood Pressure",
                    40, 250, 120
                )

            with col2:
                cholesterol = st.number_input(
                    "Cholesterol",
                    50, 500, 180
                )

                insulin = st.number_input(
                    "Insulin",
                    0, 1000, 100
                )

                hba1c = st.number_input(
                    "HbA1c",
                    3.0, 20.0, 5.5,
                    step=0.1
                )

        with tab3:

            st.markdown("### Lifestyle Information")

            col1, col2 = st.columns(2)

            with col1:
                smoking = st.selectbox(
                    "Smoking Status",
                    ["Current", "Former", "Never"]
                )

            with col2:
                activity = st.selectbox(
                    "Physical Activity",
                    ["High", "Low", "Moderate"]
                )

        st.write("")

        submitted = st.form_submit_button(
            "✦ GENERATE AI PREDICTION",
            use_container_width=True
        )

    # =================================
    # PREDICTION RESULT
    # =================================

    if submitted:

        values = {
            "Age": age,
            "BMI": bmi,
            "Glucose": glucose,
            "Blood_Pressure": bp,
            "Cholesterol": cholesterol,
            "Insulin": insulin,
            "HbA1c": hba1c,

            "Gender_Male": int(gender == "Male"),
            "Gender_Unknown": int(gender == "Unknown"),

            "Smoking_Status_Former": int(smoking == "Former"),
            "Smoking_Status_Never": int(smoking == "Never"),

            "Physical_Activity_Low": int(activity == "Low"),
            "Physical_Activity_Moderate": int(activity == "Moderate"),

            "Family_History_Yes": int(family == "Yes")
        }

        input_df = pd.DataFrame([values])

        input_df = input_df.reindex(
            columns=features,
            fill_value=0
        )

        input_df = pd.DataFrame(
            imputer.transform(input_df),
            columns=features
        )

        prediction = model.predict(input_df)[0]

        st.divider()

        st.markdown("## ✨ AI Analysis Result")

        if prediction == 1:
            result_text = "Positive Prediction"
            result_color = "#f87171"
            result_icon = "⚠️"
        else:
            result_text = "Negative Prediction"
            result_color = "#6ee7b7"
            result_icon = "✓"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-heading">
                HEALTHPREDICT ANALYSIS
            </div>
            <div class="result-value">
                {result_icon} {result_text}
            </div>
            <p style="color:#cbd5e1;">
                Machine learning classification generated
                successfully.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):

            probs = model.predict_proba(input_df)[0]

            positive_index = list(model.classes_).index(1)

            probability = probs[positive_index] * 100

            st.write("")

            st.metric(
                "Estimated Positive-Class Probability",
                f"{probability:.2f}%"
            )

            st.progress(
                int(round(probability))
            )

        with st.expander("View Patient Summary"):

            summary = pd.DataFrame({
                "Parameter": [
                    "Age", "Gender", "BMI", "Glucose",
                    "Blood Pressure", "Cholesterol",
                    "Insulin", "HbA1c", "Smoking",
                    "Physical Activity", "Family History"
                ],
                "Value": [
                    age, gender, bmi, glucose, bp,
                    cholesterol, insulin, hba1c,
                    smoking, activity, family
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            "This is an academic machine learning estimate, "
            "not a medical diagnosis. Consult a qualified "
            "healthcare professional for medical advice."
        )

# =====================================
# ABOUT PROJECT PAGE
# =====================================

elif page == "📘 About Project":

    st.markdown("""
    <div class="hero">
        <h1>📘 About HealthPredict</h1>
        <p>
        Machine Learning Based Diabetes Prediction Project
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Project Overview")

    st.markdown("""
    <div class="panel">
        <p>
        HealthPredict is an academic machine learning
        application designed to classify diabetes risk
        using patient health and lifestyle information.
        </p>
        <p>
        The application uses a trained classification
        model and preprocessing pipeline saved from
        the Jupyter Notebook.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Technology Stack")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🐍 Python")

    with col2:
        st.info("🤖 Scikit-learn")

    with col3:
        st.info("🌐 Streamlit")

    st.markdown("## Model Information")

    st.write("**Algorithm:**", type(model).__name__)

    st.write("**Input features:**", len(features))

    st.write("**Model status:** Loaded")

# =====================================
# FOOTER
# =====================================

st.markdown("""
<div class="footer">
    HEALTHPREDICT AI | DIABETES PREDICTION SYSTEM
    <br>
    Developed with Python, Scikit-learn & Streamlit
    <br>
    Academic Machine Learning Project
</div>
""", unsafe_allow_html=True)