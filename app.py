# app.py
import streamlit as st
import joblib
import numpy as np
import os
import requests
from io import BytesIO

# ----------------- Config -----------------
st.set_page_config(page_title="Lung Cancer Checker (Streamlit)", layout="centered")

# Local model filename (the repo name you mentioned: 'lung cnacer.pkl')
LOCAL_MODEL_FILENAME = "lung_cancer_model.pkl"  # NOTE: filename uses underscore; change if your actual filename differs

# If your model is hosted on GitHub raw, set the raw URL below (example):
# GITHUB_MODEL_RAW_URL = "https://raw.githubusercontent.com/<username>/<repo>/main/lung_cnacer.pkl"
GITHUB_MODEL_RAW_URL = ""  # <-- Put the raw GitHub URL here if you want the app to download the model automatically

# Path to the uploaded image (developer message: use this local path as the file URL)
PARAMETERS_IMAGE_PATH = "/mnt/data/da734b7e-2a46-40b4-bf67-b5b5adb501fc.png"

# NOTE: FEATURE_ORDER must match the order the model expects (excluding the target 'LUNG_CANCER').
# Based on the image you provided, these are the features (we exclude LUNG_CANCER):
FEATURE_ORDER = [
    "SMOKING",
    "YELLOW_FINGERS",
    "ANXIETY",
    "PEER_PRESSURE",
    "CHRONIC_DISEASE",
    "FATIGUE",
    "ALLERGY",
    "WHEEZING",
    "ALCOHOL_CONSUMING",
    "COUGHING",
    "SHORTNESS_OF_BREATH",
    "SWALLOWING_DIFFICULTY",
    "CHEST_PAIN",
    # computed feature (will be appended automatically)
    "ANXIETY_PLUS_YELLOWFINGERS"
]
# ------------------------------------------

st.title("Lung Cancer Checker (Patient / Doctor)")

# Sidebar mode selector
mode = st.sidebar.radio("Select Mode", options=["Patient", "Doctor"])

# Show the parameters image so user sees the list (developer: use the provided local path)
if os.path.exists(PARAMETERS_IMAGE_PATH):
    st.sidebar.image(PARAMETERS_IMAGE_PATH, caption="Model parameters (source image)", use_column_width=True)
else:
    st.sidebar.write("Parameters image not found locally.")

# ----------------- Model loader -----------------
@st.cache_resource
def load_model_local_or_remote(local_path: str, github_raw_url: str = ""):
    """
    Try to load the model from local_path first.
    If not present and github_raw_url is provided, try to download.
    Returns: (model_or_none, error_message_or_none)
    """
    # 1) try local
    if os.path.exists(local_path):
        try:
            m = joblib.load(local_path)
            return m, None
        except Exception as e:
            return None, f"Failed to load local model '{local_path}': {e}"

    # 2) try download from GitHub raw URL if provided
    if github_raw_url:
        try:
            resp = requests.get(github_raw_url, timeout=30)
            resp.raise_for_status()
            # load from bytes
            m = joblib.load(BytesIO(resp.content))
            # optionally save locally
            try:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
            except Exception:
                # not critical
                pass
            return m, None
        except Exception as e:
            return None, f"Failed to download model from GitHub URL: {e}"

    return None, f"Model not found at '{local_path}' and no GitHub raw URL provided."

# Load model once
model, model_err = load_model_local_or_remote(LOCAL_MODEL_FILENAME, GITHUB_MODEL_RAW_URL)

# ----------------- UI flows -----------------
if mode == "Doctor":
    st.header("Doctor mode")
    st.warning("Doctor mode is not integrated yet. Only Patient mode is active in this prototype.")
    st.write("You can use Patient mode to test model predictions. Doctor features (history, edit, authentication) will be added later.")
    st.stop()

# Patient mode
st.header("Patient mode — enter your details")

st.markdown(
    """
    Please fill in the following parameters. Most fields are binary (0 = No, 1 = Yes).
    The app will automatically compute the combined feature **ANXIETY_PLUS_YELLOWFINGERS** = 1 when both Anxiety and Yellow_Fingers are 1.
    """
)

# Helper to display binary 0/1 selection as Yes/No
def binary_select(label, key, default=0):
    choice = st.selectbox(label, options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=default, key=key)
    return int(choice)

# Render inputs for each feature except the computed one
input_values = {}
input_values["SMOKING"] = binary_select("Smoking (Do you smoke?)", "SMOKING", default=0)
input_values["YELLOW_FINGERS"] = binary_select("Yellow fingers (Do you have yellowish fingers?)", "YELLOW_FINGERS", default=0)
input_values["ANXIETY"] = binary_select("Anxiety (Do you frequently feel anxious?)", "ANXIETY", default=0)
input_values["PEER_PRESSURE"] = binary_select("Peer pressure (Do you face peer pressure?)", "PEER_PRESSURE", default=0)
input_values["CHRONIC_DISEASE"] = binary_select("Chronic disease (Do you have chronic disease?)", "CHRONIC_DISEASE", default=0)
input_values["FATIGUE"] = binary_select("Fatigue (Are you fatigued often?)", "FATIGUE", default=0)
input_values["ALLERGY"] = binary_select("Allergy (Do you have allergies?)", "ALLERGY", default=0)
input_values["WHEEZING"] = binary_select("Wheezing (Do you wheeze?)", "WHEEZING", default=0)
input_values["ALCOHOL_CONSUMING"] = binary_select("Alcohol consuming (Do you drink alcohol?)", "ALCOHOL_CONSUMING", default=0)
input_values["COUGHING"] = binary_select("Coughing (Do you cough frequently?)", "COUGHING", default=0)
input_values["SHORTNESS_OF_BREATH"] = binary_select("Shortness of breath (Do you have it?)", "SHORTNESS_OF_BREATH", default=0)
input_values["SWALLOWING_DIFFICULTY"] = binary_select("Swallowing difficulty (Do you have it?)", "SWALLOWING_DIFFICULTY", default=0)
input_values["CHEST_PAIN"] = binary_select("Chest pain (Do you have chest pain?)", "CHEST_PAIN", default=0)

# Compute the derived feature automatically
anxiety = int(input_values.get("ANXIETY", 0))
yellow = int(input_values.get("YELLOW_FINGERS", 0))
anxiety_plus_yellow = 1 if (anxiety == 1 and yellow == 1) else 0
input_values["ANXIETY_PLUS_YELLOWFINGERS"] = anxiety_plus_yellow

st.markdown("**Computed field**: ANXIETY_PLUS_YELLOWFINGERS = 1 if both Anxiety and Yellow_Fingers are Yes, else 0.")
st.write(f"Computed ANXIETY_PLUS_YELLOWFINGERS = **{anxiety_plus_yellow}**")

# Submit button
if st.button("Run model prediction"):
    if model is None:
        st.error(f"Model not loaded. {model_err}")
    else:
        # Assemble feature vector in FEATURE_ORDER
        try:
            x = [float(input_values.get(feat, 0)) for feat in FEATURE_ORDER]
            X = np.array(x).reshape(1, -1)
            # predict
            pred = int(model.predict(X)[0])
            prob = None
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[0][1])
            elif hasattr(model, "decision_function"):
                score = model.decision_function(X)[0]
                prob = 1 / (1 + np.exp(-score))  # pseudo-probability (not calibrated)
            # show results
            st.success(f"Prediction: {'Positive for lung-cancer (model class=1)' if pred == 1 else 'Negative (model class=0)'}")
            if prob is not None:
                st.info(f"Probability for class=1: {prob:.3f}")
            st.json({
                "feature_order_used": FEATURE_ORDER,
                "input_features_sent": {k: input_values.get(k) for k in FEATURE_ORDER},
                "prediction": pred,
                "probability": prob
            })
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.caption("If your model expects a different feature order or names, update FEATURE_ORDER and the inputs in this file accordingly.")
