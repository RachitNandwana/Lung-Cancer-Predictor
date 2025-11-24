# app.py
import streamlit as st
import joblib
import numpy as np
import os
import sqlite3
import pandas as pd
from io import BytesIO
from datetime import datetime

# Optional imports for huggingface hub & keras
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    tf = None
    keras_load_model = None

# ----------------- Config -----------------
st.set_page_config(page_title="Lung Cancer Checker (Streamlit)", layout="centered")

# Local model filename (if you have already downloaded)
LOCAL_MODEL_FILENAME = "lung_cnacer.pkl"  # or lung_cnacer.h5 if Keras

# If using Hugging Face: provide repo_id and filename in the sidebar (example below)
# e.g. repo_id = "username/repo_name", filename = "lung_cnacer.h5"
# If the file is public you don't need a token; for private use an HF token is required.
PARAMETERS_IMAGE_PATH = "/mnt/data/da734b7e-2a46-40b4-bf67-b5b5adb501fc.png"

DB_PATH = "predictions.db"
DOCTOR_PASSWORD = "doctor123"  # simple password for demo. Replace or implement real auth for production.

# Features expected by the model (exclude target)
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
    "ANXIETY_PLUS_YELLOWFINGERS"
]

# ----------------- DB helpers -----------------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_json TEXT,
            prediction INTEGER,
            probability REAL
        )
    """)
    conn.commit()
    return conn

DB_CONN = init_db()

def save_record(input_dict, prediction, probability):
    cur = DB_CONN.cursor()
    cur.execute(
        "INSERT INTO predictions (timestamp, input_json, prediction, probability) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), str(input_dict), int(prediction), float(probability) if probability is not None else None)
    )
    DB_CONN.commit()

def get_all_records():
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", DB_CONN)
    return df

def delete_record(record_id):
    cur = DB_CONN.cursor()
    cur.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
    DB_CONN.commit()

# ----------------- Model loader -----------------
@st.cache_resource
def load_model(path_local: str = "", hf_repo_id: str = "", hf_filename: str = "", hf_token: str = ""):
    """
    Try to load model from local path first. If not present and hf_repo_id/hf_filename provided,
    try to download from Hugging Face hub. Supports:
      - scikit-learn .pkl via joblib
      - keras .h5 via tensorflow.keras.models.load_model (requires TF)
    Returns (model, meta) where meta includes {'type': 'sklearn'|'keras', 'source': 'local'|'hf', 'path': str}
    """
    # 1) Try local file if provided and exists
    if path_local:
        if os.path.exists(path_local):
            try:
                if path_local.lower().endswith(".pkl"):
                    m = joblib.load(path_local)
                    return m, {"type": "sklearn", "source": "local", "path": path_local}
                elif path_local.lower().endswith(".h5") or path_local.lower().endswith(".keras"):
                    if keras_load_model is None:
                        return None, {"error": "TensorFlow is not installed; cannot load .h5 model locally."}
                    m = keras_load_model(path_local)
                    return m, {"type": "keras", "source": "local", "path": path_local}
                else:
                    # try joblib then keras
                    try:
                        m = joblib.load(path_local)
                        return m, {"type": "sklearn", "source": "local", "path": path_local}
                    except Exception:
                        if keras_load_model:
                            m = keras_load_model(path_local)
                            return m, {"type": "keras", "source": "local", "path": path_local}
                        raise
            except Exception as e:
                return None, {"error": f"Failed to load local model: {e}"}

    # 2) Try Hugging Face hub if repo/filename provided
    if hf_repo_id and hf_filename:
        if hf_hub_download is None:
            return None, {"error": "huggingface_hub not installed; can't download from HF hub."}
        try:
            # hf_hub_download will download to cache and return local path
            local_file = hf_hub_download(repo_id=hf_repo_id, filename=hf_filename, use_auth_token=hf_token or None)
            # load depending on extension
            if local_file.lower().endswith(".pkl"):
                m = joblib.load(local_file)
                return m, {"type": "sklearn", "source": "hf", "path": local_file}
            elif local_file.lower().endswith(".h5") or local_file.lower().endswith(".keras"):
                if keras_load_model is None:
                    return None, {"error": "TensorFlow is not installed; cannot load .h5 model from HF."}
                m = keras_load_model(local_file)
                return m, {"type": "keras", "source": "hf", "path": local_file}
            else:
                # fallback try joblib
                try:
                    m = joblib.load(local_file)
                    return m, {"type": "sklearn", "source": "hf", "path": local_file}
                except Exception:
                    if keras_load_model:
                        m = keras_load_model(local_file)
                        return m, {"type": "keras", "source": "hf", "path": local_file}
                    raise
        except Exception as e:
            return None, {"error": f"Failed to download/load from Hugging Face hub: {e}"}
    return None, {"error": "No local model found and no Hugging Face repo/filename provided."}

# ----------------- UI & logic -----------------
st.title("Lung Cancer Checker — Patient & Doctor (Streamlit)")

# show parameter image in sidebar if exists
if os.path.exists(PARAMETERS_IMAGE_PATH):
    st.sidebar.image(PARAMETERS_IMAGE_PATH, caption="Model parameters (source image)", use_column_width=True)
else:
    st.sidebar.write("Parameters image not found.")

st.sidebar.markdown("---")
st.sidebar.header("Model source (optional)")
st.sidebar.write("If your model is on Hugging Face (large files >100MB), enter repo & filename below.")
hf_repo_id = st.sidebar.text_input("HF repo id (e.g. user/repo)", value="")
hf_filename = st.sidebar.text_input("HF filename (e.g. lung_cnacer.h5 or lung_cnacer.pkl)", value="")
# token from secrets or env fallback (recommended to set securely)
hf_token = st.secrets.get("HF_TOKEN") if "HF_TOKEN" in st.secrets else os.environ.get("HF_TOKEN", "")
if hf_token:
    st.sidebar.write("HF token: using configured token.")
else:
    st.sidebar.info("If model is private, set HF token via Streamlit secrets or HF_TOKEN env var.")

# allow overriding local filename if your local copy differs
local_model_override = st.sidebar.text_input("Local model filename (optional)", value=LOCAL_MODEL_FILENAME)

st.sidebar.markdown("---")
mode = st.sidebar.radio("Select mode", ["Patient", "Doctor"])

# Load model (cached resource)
model, model_meta = load_model(path_local=local_model_override.strip() or None,
                              hf_repo_id=hf_repo_id.strip(),
                              hf_filename=hf_filename.strip(),
                              hf_token=hf_token)

if model is None and "error" in model_meta:
    st.sidebar.error(model_meta["error"])
elif model is not None:
    st.sidebar.success(f"Model loaded ({model_meta.get('type')}, from {model_meta.get('source')})")

# helper to render input fields
def binary_select(label, key, default=0):
    choice = st.selectbox(label, options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=default, key=key)
    return int(choice)

def assemble_inputs(prefix="p_"):
    inputs = {}
    inputs["SMOKING"] = binary_select("Smoking", prefix + "SMOKING", 0)
    inputs["YELLOW_FINGERS"] = binary_select("Yellow fingers", prefix + "YELLOW_FINGERS", 0)
    inputs["ANXIETY"] = binary_select("Anxiety", prefix + "ANXIETY", 0)
    inputs["PEER_PRESSURE"] = binary_select("Peer pressure", prefix + "PEER_PRESSURE", 0)
    inputs["CHRONIC_DISEASE"] = binary_select("Chronic disease", prefix + "CHRONIC_DISEASE", 0)
    inputs["FATIGUE"] = binary_select("Fatigue", prefix + "FATIGUE", 0)
    inputs["ALLERGY"] = binary_select("Allergy", prefix + "ALLERGY", 0)
    inputs["WHEEZING"] = binary_select("Wheezing", prefix + "WHEEZING", 0)
    inputs["ALCOHOL_CONSUMING"] = binary_select("Alcohol consuming", prefix + "ALCOHOL_CONSUMING", 0)
    inputs["COUGHING"] = binary_select("Coughing", prefix + "COUGHING", 0)
    inputs["SHORTNESS_OF_BREATH"] = binary_select("Shortness of breath", prefix + "SHORTNESS_OF_BREATH", 0)
    inputs["SWALLOWING_DIFFICULTY"] = binary_select("Swallowing difficulty", prefix + "SWALLOWING_DIFFICULTY", 0)
    inputs["CHEST_PAIN"] = binary_select("Chest pain", prefix + "CHEST_PAIN", 0)
    # computed feature
    inputs["ANXIETY_PLUS_YELLOWFINGERS"] = 1 if (inputs["ANXIETY"] == 1 and inputs["YELLOW_FINGERS"] == 1) else 0
    return inputs

# common predict function handling sklearn or keras
def predict_with_model(model_obj, model_meta, input_dict):
    # assemble features in right order
    x = np.array([float(input_dict.get(f, 0)) for f in FEATURE_ORDER]).reshape(1, -1)
    mtype = model_meta.get("type") if model_meta else None
    if mtype == "sklearn" or hasattr(model_obj, "predict_proba") or hasattr(model_obj, "predict"):
        # sklearn-style
        pred = int(model_obj.predict(x)[0])
        prob = None
        if hasattr(model_obj, "predict_proba"):
            prob = float(model_obj.predict_proba(x)[0][1])
        elif hasattr(model_obj, "decision_function"):
            score = model_obj.decision_function(x)[0]
            prob = 1 / (1 + np.exp(-score))
        return pred, prob
    else:
        # assume keras: model.predict -> probabilities or logits
        if keras_load_model is None:
            raise RuntimeError("TensorFlow not installed; cannot run Keras model.")
        y = model_obj.predict(x)
        # y shape may be (1,1) or (1,2). We'll interpret:
        if y.ndim == 2 and y.shape[1] == 2:
            prob = float(y[0][1])
            pred = int(np.argmax(y[0]))
        else:
            # single output between 0-1
            prob = float(y[0][0])
            pred = 1 if prob >= 0.5 else 0
        return pred, prob

# ----------------- Patient mode -----------------
if mode == "Patient":
    st.header("Patient mode — Enter your details")
    st.write("Fill the form and click **Predict**. The app computes ANXIETY_PLUS_YELLOWFINGERS automatically.")
    with st.form("patient_form"):
        patient_inputs = assemble_inputs(prefix="p_")
        submitted = st.form_submit_button("Predict")
    if submitted:
        if model is None:
            st.error("Model not loaded. Configure a local filename or Hugging Face repo/filename in the sidebar.")
        else:
            try:
                pred, prob = predict_with_model(model, model_meta, patient_inputs)
                st.success(f"Prediction: {'Positive (model class=1)' if pred == 1 else 'Negative (model class=0)'}")
                if prob is not None:
                    st.info(f"Probability (class=1): {prob:.3f}")
                st.json({"features_sent": {k: patient_inputs[k] for k in FEATURE_ORDER},
                         "prediction": int(pred),
                         "probability": prob})
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ----------------- Doctor mode -----------------
elif mode == "Doctor":
    st.header("Doctor mode — Protected")
    # simple password gate for demo
    pwd = st.text_input("Enter doctor password", type="password")
    if pwd != DOCTOR_PASSWORD:
        st.warning("Doctor mode requires a password. (Demo only: use correct password to proceed.)")
        st.stop()

    st.success("Authenticated as doctor (demo). You can run predictions, save them to history, view/delete/export records.")

    # doctor can run prediction on a patient
    with st.expander("Run prediction for a patient (enter details)"):
        with st.form("doctor_predict_form"):
            doc_inputs = assemble_inputs(prefix="d_")
            save_after = st.checkbox("Save this prediction to history", value=True)
            submit_doc = st.form_submit_button("Run & (optionally) Save")
        if submit_doc:
            if model is None:
                st.error("Model not loaded. Configure model in the sidebar.")
            else:
                try:
                    pred, prob = predict_with_model(model, model_meta, doc_inputs)
                    st.info(f"Prediction: {'Positive' if pred == 1 else 'Negative'} — Probability: {prob:.3f}" if prob is not None else f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
                    if save_after:
                        save_record({k: doc_inputs[k] for k in FEATURE_ORDER}, pred, prob)
                        st.success("Prediction saved to history.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    st.markdown("---")
    st.subheader("Saved Predictions (History)")
    df = get_all_records()
    st.write(f"Total records: {len(df)}")
    if len(df) == 0:
        st.info("No saved predictions yet.")
    else:
        # show table
        st.dataframe(df)

        # allow deleting a record
        with st.form("delete_form"):
            del_id = st.number_input("Enter record id to delete", min_value=1, value=0, step=1)
            del_submit = st.form_submit_button("Delete record")
        if del_submit and del_id > 0:
            delete_record(int(del_id))
            st.experimental_rerun()

        # export CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV of records", data=csv, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: For production use, replace demo password with proper authentication, secure the DB, and validate/clean inputs.")
