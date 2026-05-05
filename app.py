import streamlit as st
from disease_prediction import predict_diabetes_proba, predict_heart_proba
import time

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Health Predictor", layout="wide")

# ------------------ TITLE ------------------
st.markdown("<h1 style='text-align:center;'>🩺 AI Health Prediction System</h1>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("🔍 Navigation")
option = st.sidebar.selectbox("Choose Disease", ["Diabetes", "Heart Disease"])


# =========================================================
# 🔷 FUNCTION: RISK CLASSIFICATION
# =========================================================
def get_risk_label(prob):
    if prob >= 70:
        return "High"
    elif prob >= 40:
        return "Moderate"
    else:
        return "Low"


# =========================================================
# 🔷 DIABETES
# =========================================================
if option == "Diabetes":

    st.header("🧪 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        preg = st.slider("Pregnancies", 0, 15, 1)
        glucose = st.slider("Glucose", 50, 200, 100)
        bp = st.slider("Blood Pressure", 40, 140, 70)
        skin = st.slider("Skin Thickness", 0, 60, 20)

    with col2:
        insulin = st.slider("Insulin", 0, 300, 80)
        bmi = st.slider("BMI", 10, 50, 25)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 10, 100, 30)

    if st.button("🔍 Predict Diabetes"):

        data = [preg, glucose, bp, skin, insulin, bmi, dpf, age]

        with st.spinner("Analyzing..."):
            time.sleep(2)

        risk_prob = predict_diabetes_proba(data) * 100
        risk_label = get_risk_label(risk_prob)

        st.subheader("🧾 Medical Report")

        # Result message based on probability
        if risk_prob >= 50:
            st.error("⚠️ Diabetes Risk Detected")
        else:
            st.success("✅ Low Diabetes Risk")

        # Risk display
        st.markdown(f"### 🩺 Risk Level: **{risk_label} ({risk_prob:.2f}%)**")
        st.progress(int(risk_prob))

        # Confidence explanation
        st.info("📊 This percentage represents the model's confidence in the prediction.")

        # Feedback
        if risk_label == "High":
            st.markdown("""
            #### 📋 Consultant Feedback:
            - High glucose or BMI may indicate diabetes risk.

            #### 💡 Advice:
            - Follow low sugar diet  
            - Exercise daily  
            - Monitor glucose levels  
            - Consult a doctor  
            """)
        elif risk_label == "Moderate":
            st.markdown("""
            #### 📋 Consultant Feedback:
            - Some indicators are slightly elevated.

            #### 💡 Advice:
            - Improve diet  
            - Increase physical activity  
            - Regular monitoring  
            """)
        else:
            st.markdown("""
            #### 📋 Consultant Feedback:
            - Your health indicators are within normal range.

            #### 💡 Advice:
            - Maintain healthy lifestyle  
            - Regular checkups  
            """)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Glucose", glucose)
        col2.metric("BMI", bmi)
        col3.metric("Age", age)


# =========================================================
# 🔷 HEART
# =========================================================
elif option == "Heart Disease":

    st.header("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 20, 100, 40)
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.slider("Chest Pain Type", 0, 3, 1)
        trestbps = st.slider("Blood Pressure", 80, 200, 120)

    with col2:
        chol = st.slider("Cholesterol", 100, 400, 200)
        fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
        restecg = st.slider("Rest ECG", 0, 2, 1)
        thalach = st.slider("Max Heart Rate", 60, 220, 150)

    with col3:
        exang = st.selectbox("Exercise Angina", [0, 1])
        oldpeak = st.slider("Oldpeak", 0, 6, 1)
        slope = st.slider("Slope", 0, 2, 1)
        ca = st.slider("Major Vessels", 0, 4, 0)
        thal = st.slider("Thal", 1, 3, 2)

    if st.button("🔍 Predict Heart Disease"):

        data = [age, sex, cp, trestbps, chol, fbs,
                restecg, thalach, exang, oldpeak,
                slope, ca, thal]

        with st.spinner("Analyzing..."):
            time.sleep(2)

        risk_prob = predict_heart_proba(data) * 100
        risk_label = get_risk_label(risk_prob)

        st.subheader("🧾 Medical Report")

        # Result message
        if risk_prob >= 50:
            st.error("⚠️ Heart Disease Risk Detected")
        else:
            st.success("✅ Low Heart Disease Risk")

        # Risk display
        st.markdown(f"### ❤️ Risk Level: **{risk_label} ({risk_prob:.2f}%)**")
        st.progress(int(risk_prob))

        # Confidence explanation
        st.info("📊 This percentage shows how strongly the model predicts heart disease risk.")

        # Feedback
        if risk_label == "High":
            st.markdown("""
            #### 📋 Consultant Feedback:
            - High BP or cholesterol may indicate heart risk.

            #### 💡 Advice:
            - Reduce salt & fat intake  
            - Exercise regularly  
            - Avoid smoking  
            - Consult a cardiologist  
            """)
        elif risk_label == "Moderate":
            st.markdown("""
            #### 📋 Consultant Feedback:
            - Some risk factors are present.

            #### 💡 Advice:
            - Improve lifestyle  
            - Monitor BP & cholesterol  
            """)
        else:
            st.markdown("""
            #### 📋 Consultant Feedback:
            - Heart health indicators are normal.

            #### 💡 Advice:
            - Maintain healthy lifestyle  
            """)

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Age", age)
        col2.metric("Cholesterol", chol)
        col3.metric("BP", trestbps)