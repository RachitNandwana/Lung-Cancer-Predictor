# app.py
import streamlit as st
import requests
import numpy as np
import pandas as pd
import sqlite3
import os
import json
from datetime import datetime
from typing import Any, Dict, List

# ----------------- Configuration -----------------
st.set_page_config(page_title="Lung Cancer Checker (Streamlit)", layout="centered")

# Path to parameter image that you uploaded (already present in environment)
PARAMETERS_IMAGE_PATH = "/mnt/data/da734b7e-2a46-40b4-bf67-b5b5adb501fc.png"

# Feature order expected by the model (must match model training)
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

# SQLite DB for doctor history
DB_PATH = "predictions.db"

# ----------------- Helpers: DB -----------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_json TEXT,
            prediction INTEGER,
            probability REAL
        )
        """
    )
    conn.commit()
    return conn

DB_CONN = init_db()

def save_record(input_dict: Dict[str, Any], prediction: int, probability: float):
    cur = DB_CONN.cursor()
    cur.execute(
        "INSERT INTO predictions (timestamp, input_json, prediction, probability) VALUES (?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), json.dumps(input_dict), int(prediction), float(probability) if probability is not None else None)
    )
    DB_CONN.commit()

def get_all_records() -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", DB_CONN)
    return df

def delete_record(record_id: int):
    cur = DB_CONN.cursor()
    cur.execute("DELETE FROM predictions WHERE id = ?", (int(record_id),))
    DB_CONN.commit()

# ----------------- Helpers: UI inputs -----------------
def binary_select(label: str, key: str, default: int = 0) -> int:
    """Render a Yes/No select and return 0/1."""
    return int(st.selectbox(label, options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=default, key=key))

def assemble_inputs(prefix: str) -> Dict[str, int]:
    """Render inputs for all features (except computed) and compute the derived feature."""
    inputs = {}
    # Render fields in same order used previously (excluding the computed field)
    inputs["SMOKING"] = binary_select("Smoking (Do you smoke?)", prefix + "SMOKING")
    inputs["YELLOW_FINGERS"] = binary_select("Yellow fingers (Do you have yellowish fingers?)", prefix + "YELLOW_FINGERS")
    inputs["ANXIETY"] = binary_select("Anxiety (Do you frequently feel anxious?)", prefix + "ANXIETY")
    inputs["PEER_PRESSURE"] = binary_select("Peer pressure (Do you face peer pressure?)", prefix + "PEER_PRESSURE")
    inputs["CHRONIC_DISEASE"] = binary_select("Chronic disease (Do you have chronic disease?)", prefix + "CHRONIC_DISEASE")
    inputs["FATIGUE"] = binary_select("Fatigue (Are you fatigued often?)", prefix + "FATIGUE")
    inputs["ALLERGY"] = binary_select("Allergy (Do you have allergies?)", prefix + "ALLERGY")
    inputs["WHEEZING"] = binary_select("Wheezing (Do you wheeze?)", prefix + "WHEEZING")
    inputs["ALCOHOL_CONSUMING"] = binary_select("Alcohol consuming (Do you drink alcohol?)", prefix + "ALCOHOL_CONSUMING")
    inputs["COUGHING"] = binary_select("Coughing (Do you cough frequently?)", prefix + "COUGHING")
    inputs["SHORTNESS_OF_BREATH"] = binary_select("Shortness of breath (Do you have it?)", prefix + "SHORTNESS_OF_BREATH")
    inputs["SWALLOWING_DIFFICULTY"] = binary_select("Swallowing difficulty (Do you have it?)", prefix + "SWALLOWING_DIFFICULTY")
    inputs["CHEST_PAIN"] = binary_select("Chest pain (Do you have chest pain?)", prefix + "CHEST_PAIN")

    # Derived feature
    inputs["ANXIETY_PLUS_YELLOWFINGERS"] = 1 if (inputs["ANXIETY"] == 1 and inputs["YELLOW_FINGERS"] == 1) else 0
    return inputs

# ----------------- Helpers: Remote inference -----------------
# 1) Put these two secrets in Streamlit Cloud (Settings → Secrets):
#    HF_INFERENCE_URL  (e.g. https://api-inference.huggingface.co/models/username/repo)
#    HF_TOKEN          (only if your model/prediction endpoint requires authentication)
HF_INFERENCE_URL = st.secrets.get("HF_INFERENCE_URL") or os.environ.get("HF_INFERENCE_URL")
HF_TOKEN = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
# Optional DOCTOR_PASSWORD secret override
DOCTOR_PASSWORD_SECRET = st.secrets.get("DOCTOR_PASSWORD") or os.environ.get("DOCTOR_PASSWORD", "doctor123")

def call_hf_inference(feature_vector: List[int]) -> Any:
    """
    Call remote inference endpoint. By default uses HF Inference API style endpoint.
    Payload: {"inputs": feature_vector}
    Returns the JSON response from the remote endpoint.
    """
    if not HF_INFERENCE_URL:
        raise RuntimeError("HF_INFERENCE_URL is not configured. Set the secret on Streamlit Cloud.")
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    payload = {"inputs": feature_vector}
    res = requests.post(HF_INFERENCE_URL, headers=headers, json=payload, timeout=120)
    res.raise_for_status()
    return res.json()

def parse_remote_result(result_json: Any) -> Dict[str, Any]:
    """
    Attempt to parse common HF/endpoint responses into {'prediction': 0|1, 'probability': float|None}.
    The remote API may return:
      - list of dicts [{'label': '0' or '1', 'score': 0.7}, ...]
      - dict with {'label': 'LABEL', 'score': ...}
      - dict with {'pred': 0, 'prob': 0.7} (custom)
      - any structure: we try to find numeric score/label heuristically
    """
    pred = None
    prob = None

    # If result is a list with label/score
    try:
        if isinstance(result_json, list) and len(result_json) > 0 and isinstance(result_json[0], dict):
            # choose the best entry if multiple
            entry = result_json[0]
            # label may be 'LABEL_0' or 'NEGATIVE' or '0'
            if "label" in entry:
                label = entry.get("label")
                # try find numeric in label
                try:
                    pred = int("".join(ch for ch in str(label) if ch.isdigit()))
                except Exception:
                    # if label is 'POSITIVE' or 'NEGATIVE'
                    label_str = str(label).lower()
                    if "pos" in label_str or "positive" in label_str:
                        pred = 1
                    elif "neg" in label_str or "negative" in label_str:
                        pred = 0
            if "score" in entry:
                prob = float(entry.get("score"))
            # fallback: if no label but 'score' present and only one output, interpret score threshold 0.5
            if pred is None and prob is not None:
                pred = 1 if prob >= 0.5 else 0
            return {"prediction": pred, "probability": prob}
    except Exception:
        pass

    # If result is a dict
    if isinstance(result_json, dict):
        # common custom shapes
        if "pred" in result_json and "prob" in result_json:
            try:
                pred = int(result_json["pred"])
                prob = float(result_json["prob"])
                return {"prediction": pred, "probability": prob}
            except Exception:
                pass
        if "prediction" in result_json and "probability" in result_json:
            try:
                return {"prediction": int(result_json["prediction"]), "probability": float(result_json["probability"])}
            except Exception:
                pass
        # If label & score top-level
        if "label" in result_json and "score" in result_json:
            label = result_json["label"]
            try:
                pred = int("".join(ch for ch in str(label) if ch.isdigit()))
            except Exception:
                label_str = str(label).lower()
                if "pos" in label_str:
                    pred = 1
                elif "neg" in label_str:
                    pred = 0
            prob = float(result_json["score"])
            if pred is None and prob is not None:
                pred = 1 if prob >= 0.5 else 0
            return {"prediction": pred, "probability": prob}

    # As a last resort, try to find any float in the returned JSON and treat as prob
    try:
        text = json.dumps(result_json)
        # find numbers like 0.123
        import re
        m = re.search(r"0?\.\d+", text)
        if m:
            prob = float(m.group(0))
            pred = 1 if prob >= 0.5 else 0
            return {"prediction": pred, "probability": prob}
    except Exception:
        pass

    # Unknown structure: return the raw result in 'raw' and prediction as None
    return {"prediction": None, "probability": None, "raw": result_json}

# ----------------- App UI -----------------
st.title("Lung Cancer Checker — Patient & Doctor (Streamlit)")

if os.path.exists(PARAMETERS_IMAGE_PATH):
    st.sidebar.image(PARAMETERS_IMAGE_PATH, caption="Model parameters (source image)", use_column_width=True)
else:
    st.sidebar.write("Parameters image not found.")

st.sidebar.markdown("---")
st.sidebar.write("Remote inference endpoint:" if HF_INFERENCE_URL else "No HF inference URL set. Set HF_INFERENCE_URL in Streamlit secrets.")
if HF_INFERENCE_URL:
    st.sidebar.write(HF_INFERENCE_URL if len(HF_INFERENCE_URL) < 60 else HF_INFERENCE_URL[:60] + "...")
st.sidebar.markdown("---")

mode = st.sidebar.radio("Select mode", ["Patient", "Doctor"])

# ----------------- Patient mode -----------------
if mode == "Patient":
    st.header("Patient mode — Enter your details")
    st.write("Fill the form and press **Predict**. The derived feature `ANXIETY_PLUS_YELLOWFINGERS` is computed automatically.")
    with st.form("patient_form"):
        patient_inputs = assemble_inputs(prefix="p_")
        submitted = st.form_submit_button("Predict")
    if submitted:
        # Prepare feature vector in FEATURE_ORDER
        feature_list = [int(patient_inputs.get(f, 0)) for f in FEATURE_ORDER]
        st.write("Feature vector sent (in model feature order):")
        st.json({f: patient_inputs.get(f, 0) for f in FEATURE_ORDER})
        try:
            result = call_hf_inference(feature_list)
            parsed = parse_remote_result(result)
            # Show results
            if parsed.get("prediction") is not None:
                pred_text = "Positive (model class=1)" if parsed["prediction"] == 1 else "Negative (model class=0)"
                st.success(f"Prediction: {pred_text}")
            else:
                st.info("Prediction could not be parsed from remote response; see raw response below.")
            if parsed.get("probability") is not None:
                st.info(f"Probability (class=1): {parsed['probability']:.4f}")
            st.subheader("Remote (raw) response")
            st.json(result)
        except Exception as e:
            st.error(f"Error calling remote inference: {e}")

# ----------------- Doctor mode -----------------
elif mode == "Doctor":
    st.header("Doctor mode — (protected)")
    # Simple password gate (demo). For production use proper auth.
    supplied_pwd = st.text_input("Doctor password", type="password")
    expected_pwd = DOCTOR_PASSWORD_SECRET or "doctor123"
    if supplied_pwd != expected_pwd:
        st.warning("Doctor mode requires a password. Provide the correct password to proceed.")
        st.stop()

    st.success("Authenticated (demo). You may run predictions and manage saved records.")

    # Run a prediction for a patient
    with st.form("doctor_predict"):
        doc_inputs = assemble_inputs(prefix="d_")
        save_after = st.checkbox("Save this prediction to history", value=True)
        run = st.form_submit_button("Run prediction")
    if run:
        feature_list = [int(doc_inputs.get(f, 0)) for f in FEATURE_ORDER]
        st.write("Feature vector sent:")
        st.json({f: doc_inputs.get(f, 0) for f in FEATURE_ORDER})
        try:
            result = call_hf_inference(feature_list)
            parsed = parse_remote_result(result)
            if parsed.get("prediction") is not None:
                st.info(f"Prediction: {'Positive' if parsed['prediction'] == 1 else 'Negative'}")
            if parsed.get("probability") is not None:
                st.info(f"Probability (class=1): {parsed['probability']:.4f}")
            st.subheader("Remote (raw) response")
            st.json(result)
            if save_after:
                # save using the parsed prediction/prob (or None)
                save_record({k: doc_inputs.get(k, 0) for k in FEATURE_ORDER}, parsed.get("prediction") if parsed.get("prediction") is not None else -1, parsed.get("probability"))
                st.success("Saved record to history.")
        except Exception as e:
            st.error(f"Error calling remote inference: {e}")

    st.markdown("---")
    st.subheader("Saved predictions (history)")
    df = get_all_records()
    st.write(f"Total records: {len(df)}")
    if len(df) == 0:
        st.info("No saved predictions yet.")
    else:
        st.dataframe(df)
        # Delete a record
        with st.form("delete_record"):
            del_id = st.number_input("Enter record id to delete", min_value=0, value=0, step=1)
            del_submit = st.form_submit_button("Delete record")
        if del_submit and del_id > 0:
            delete_record(del_id)
            st.success(f"Deleted record {del_id}.")
            st.experimental_rerun()

        # Export CSV
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

st.markdown("---")
st.caption("Notes: This app calls a remote inference endpoint (Hugging Face Inference API or other). "
           "For production, replace the simple password with secure authentication and secure the database.")
