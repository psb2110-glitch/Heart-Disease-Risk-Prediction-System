import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "heart_disease_model.pkl"
DATA_PATH = BASE_DIR / "data" / "heart_data.csv"
FEATURE_INFO_PATH = BASE_DIR / "feature_info.json"

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 50%, #f8fafc 100%);
}

.main-title {
    font-size: 54px;
    font-weight: 900;
    color: #0f172a;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 24px;
    color: #2563eb;
    font-weight: 700;
}

.description {
    font-size: 18px;
    color: #475569;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 26px;
    border-radius: 20px;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    border: 1px solid #e2e8f0;
    transition: 0.3s;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.12);
}

.metric-card {
    background: white;
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.12);
    border-left: 6px solid #2563eb;
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-card h2 {
    color: #0f172a;
    font-size: 36px;
    margin-bottom: 5px;
}

.metric-card p {
    color: #64748b;
    font-size: 17px;
}

.low-risk {
    background: #dcfce7;
    color: #166534;
    padding: 24px;
    border-radius: 18px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
}

.moderate-risk {
    background: #fef3c7;
    color: #92400e;
    padding: 24px;
    border-radius: 18px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
}

.high-risk {
    background: #fee2e2;
    color: #991b1b;
    padding: 24px;
    border-radius: 18px;
    font-size: 26px;
    font-weight: 800;
    text-align: center;
}

.stButton>button {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.9rem;
    font-size: 19px;
    font-weight: 800;
}

.stButton>button:hover {
    background: linear-gradient(135deg, #1d4ed8, #0891b2);
    color: white;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_feature_info():
    with open(FEATURE_INFO_PATH, "r") as file:
        return json.load(file)


model = load_model()
data = load_data()
feature_info = load_feature_info()

st.markdown('<div class="main-title">❤️ Heart Disease Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Clinical Risk Assessment</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="description">Predict heart disease risk using machine learning and clinical health parameters with instant risk-level interpretation.</div>',
    unsafe_allow_html=True
)

with st.expander("Important note"):
    st.warning(
        "This app is created for academic and project purposes only. "
        "It is not a medical diagnosis tool. Please consult a qualified doctor for medical advice."
    )

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <h2>📊 {data.shape[0]}</h2>
        <p>Patient Records</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <h2>🧬 {data.shape[1] - 1}</h2>
        <p>Clinical Features</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card">
        <h2>🤖 ML</h2>
        <p>Prediction Model</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

left, right = st.columns([1.25, 1])

with left:
    st.markdown("## 🧾 Enter Patient Details")

    st.markdown("### 👤 Personal Information")
    c1, c2 = st.columns(2)

    with c1:
        age = st.slider("Age", 20, 90, 52)
    with c2:
        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

    st.markdown("### ❤️ Heart Parameters")
    c3, c4 = st.columns(2)

    with c3:
        cp = st.selectbox(
            "Chest Pain Type",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Typical Angina",
                1: "Atypical Angina",
                2: "Non-anginal Pain",
                3: "Asymptomatic"
            }[x]
        )
        trestbps = st.slider("Resting Blood Pressure", 80, 220, 120)
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 230, 150)

    with c4:
        chol = st.slider("Cholesterol", 100, 600, 240)
        oldpeak = st.slider("Oldpeak / ST Depression", 0.0, 7.0, 1.0, 0.1)
        exang = st.selectbox(
            "Exercise Induced Angina",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )

    st.markdown("### 🩺 Clinical Test Results")
    c5, c6 = st.columns(2)

    with c5:
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        restecg = st.selectbox("Resting ECG Result", options=[0, 1, 2])
        slope = st.selectbox("Slope", options=[0, 1, 2])

    with c6:
        ca = st.selectbox("Number of Major Vessels", options=[0, 1, 2, 3, 4])
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3])

input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

with right:
    st.markdown("## 📊 Prediction Result")

    if st.button("❤️ Analyze Heart Health", use_container_width=True):
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1]) * 100

        if probability < 40:
            risk_level = "Low Risk"
            st.markdown(f'<div class="low-risk">🟢 {risk_level}<br>{probability:.2f}%</div>', unsafe_allow_html=True)
        elif probability < 70:
            risk_level = "Moderate Risk"
            st.markdown(f'<div class="moderate-risk">🟡 {risk_level}<br>{probability:.2f}%</div>', unsafe_allow_html=True)
        else:
            risk_level = "High Risk"
            st.markdown(f'<div class="high-risk">🔴 {risk_level}<br>{probability:.2f}%</div>', unsafe_allow_html=True)

        st.progress(min(int(probability), 100))

        if prediction == 1:
            st.error("The model predicts a higher possibility of heart disease.")
        else:
            st.success("The model predicts a lower possibility of heart disease.")

        st.markdown("### 🩺 Health Suggestions")

        suggestions = []

        if trestbps >= 140:
            suggestions.append("Monitor blood pressure regularly.")
        if chol >= 240:
            suggestions.append("Reduce high-fat food intake and maintain a balanced diet.")
        if thalach < 120:
            suggestions.append("Discuss low maximum heart rate with a healthcare professional.")
        if exang == 1:
            suggestions.append("Exercise-induced angina is present. Medical consultation is recommended.")
        if oldpeak >= 2.0:
            suggestions.append("Elevated oldpeak value may need further ECG-related evaluation.")

        if suggestions:
            for item in suggestions:
                st.write(f"✅ {item}")
        else:
            st.write("✅ Maintain regular exercise, balanced diet, proper sleep, and routine checkups.")
    else:
        st.info("Fill the patient details and click Analyze Heart Health.")

st.divider()

tab1, tab2, tab3 = st.tabs(["📌 Input Summary", "📁 Dataset Summary", "📘 Feature Guide"])

with tab1:
    st.dataframe(input_df, use_container_width=True)

with tab2:
    st.write(f"Rows: **{data.shape[0]}**")
    st.write(f"Input Features: **{data.shape[1] - 1}**")
    st.write(f"Target Classes: **{sorted(data['target'].unique().tolist())}**")
    st.dataframe(data.head(), use_container_width=True)

with tab3:
    feature_table = pd.DataFrame(
        [{"Feature": key, "Meaning": value} for key, value in feature_info.items()]
    )
    st.dataframe(feature_table, use_container_width=True)

