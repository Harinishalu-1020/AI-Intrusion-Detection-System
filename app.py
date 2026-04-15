import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    s1 = joblib.load('stage1_binary_model.pkl')
    s2 = joblib.load('stage2_specialist_model.pkl')
    le = joblib.load('label_encoder.pkl')
    # Load the samples we saved
    X_sample = pd.read_csv('X_sample.csv').values
    y_sample = pd.read_csv('y_sample.csv').values.flatten()
    feat_names = pd.read_csv('features_sample.csv')
    return s1, s2, le, X_sample, y_sample, feat_names

s1, s2, le, X_test_scaled, y_test, X_orig = load_assets()

attack_classes = [cls for cls in le.classes_ if cls != 'Normal']
analysis_idx = attack_classes.index('Analysis')
backdoor_idx = attack_classes.index('Backdoor')

risk_mapping = {
    'Normal': {'score': 0, 'level': 'Safe', 'action': 'No action.'},
    'Analysis': {'score': 9, 'level': 'Critical', 'action': 'Deep Packet Inspection.'},
    'Backdoor': {'score': 10, 'level': 'Critical', 'action': 'Forensic Audit.'},
    'DoS': {'score': 7, 'level': 'High', 'action': 'DDoS Mitigation.'},
    'Exploits': {'score': 8, 'level': 'High', 'action': 'Patch Vulnerability.'},
    'Fuzzers': {'score': 4, 'level': 'Low', 'action': 'Rate Limit.'},
    'Generic': {'score': 2, 'level': 'Very Low', 'action': 'Log and Monitor.'},
    'Reconnaissance': {'score': 5, 'level': 'Medium', 'action': 'Block Source IP.'},
    'Shellcode': {'score': 9, 'level': 'Critical', 'action': 'Quarantine Host.'},
    'Worms': {'score': 10, 'level': 'Critical', 'action': 'Isolate Segment.'}
}

# --- GUI ---
st.set_page_config(page_title="AI-IDS Portfolio", layout="wide")
st.title("🛡️ Advanced Intrusion Detection System")
st.sidebar.header("Investigation Panel")

sample_id = st.sidebar.slider("Select Packet ID to Audit", 0, 99, 0)

input_row = X_test_scaled[sample_id].reshape(1, -1)
actual_label = le.classes_[y_test[sample_id]]

# Stage 1
is_attack = s1.predict(input_row)[0]

if is_attack == 0:
    st.success(f"### STATUS: CLEAN TRAFFIC (Actual: {actual_label})")
else:
    probs = s2.predict_proba(input_row)
    if probs[0][analysis_idx] > 0.25: pred_idx = analysis_idx
    elif probs[0][backdoor_idx] > 0.20: pred_idx = backdoor_idx
    else: pred_idx = np.argmax(probs)
    
    detected_name = attack_classes[pred_idx]
    risk = risk_mapping[detected_name]

    st.error(f"### ALERT: {detected_name.upper()} DETECTED")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{risk['score']}/10")
    col2.metric("Actual Threat", actual_label)
    col3.metric("AI Confidence", f"{np.max(probs)*100:.1f}%")
    
    st.info(f"**Action:** {risk['action']}")

    st.subheader("🔍 Forensic Evidence (Explainable AI)")
    explainer = shap.TreeExplainer(s2)
    shap_values = explainer.shap_values(input_row, check_additivity=False)
    
    plt.clf()
    # Logic to handle different SHAP output formats
    if isinstance(shap_values, list):
        sv = shap_values[pred_idx][0]
    else:
        sv = shap_values[0, :, pred_idx]
        
    shap.force_plot(explainer.expected_value[pred_idx], sv, feature_names=X_orig.columns, matplotlib=True, show=False)
    st.pyplot(plt.gcf())
