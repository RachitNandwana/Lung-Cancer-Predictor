import streamlit as st
import joblib
import numpy as np

model = joblib.load("lung_cancer_model.pkl")

st.title("Lung Cancer Prediction System")
st.write("Answer Yes/No to symptoms")

questions = {
    "SMOKING": st.selectbox("Do you smoke?", ["No", "Yes"]),
    "YELLOW_FINGERS": st.selectbox("Do you have yellow fingers?", ["No", "Yes"]),
    "ANXIETY": st.selectbox("Do you have anxiety?", ["No", "Yes"]),
    "PEER_PRESSURE": st.selectbox("Peer pressure present?", ["No", "Yes"]),
    "CHRONIC DISEASE": st.selectbox("Do you have chronic disease?", ["No", "Yes"]),
    "FATIGUE": st.selectbox("Do you feel fatigue?", ["No", "Yes"]),
    "ALLERGY": st.selectbox("Do you have allergy?", ["No", "Yes"]),
    "WHEEZING": st.selectbox("Do you wheeze while breathing?", ["No", "Yes"]),
    "ALCOHOL CONSUMING": st.selectbox("Do you consume alcohol?", ["No", "Yes"]),
    "COUGHING": st.selectbox("Do you cough continuously?", ["No", "Yes"]),
    "SHORTNESS OF BREATH": st.selectbox("Shortness of breath?", ["No", "Yes"]),
    "SWALLOWING DIFFICULTY": st.selectbox("Difficulty swallowing?", ["No", "Yes"]),
    "CHEST PAIN": st.selectbox("Do you have chest pain?", ["No", "Yes"]),
}

# Interaction feature similar to your training
anxiety = 1 if questions["ANXIETY"] == "Yes" else 0
yellow = 1 if questions["YELLOW_FINGERS"] == "Yes" else 0
interaction = anxiety * yellow  

vals = [1 if v=="Yes" else 0 for v in questions.values()]
vals.append(interaction)

vals = np.array(vals).reshape(1, -1)

if st.button("Predict"):
    pred = model.predict(vals)[0]
    
    if pred == 1:
        st.error("⚠️ **High Risk of Lung Cancer** — please consult a doctor")
    else:
        st.success("✅ **Low Risk** — seems safe, but stay healthy!")
