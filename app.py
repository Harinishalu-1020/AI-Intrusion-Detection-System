import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- 1. SETTINGS & ASSET LOADING ---
st.set_page_config(page_title="AI-IDS Security Dashboard", layout="wide")

@st.cache_resource
def load_all_assets():
    # Load your trained models and encoder
    s1 = joblib.load('stage1_binary_model.pkl')
    s2 = joblib.load('stage2_specialist_model.pkl')
    le = joblib.load('label_encoder.pkl')
    
    # Load the samples generated from your notebook
    X_scaled_data = pd.read_csv('X_sample.csv').values
    y_actual_labels = pd.read_csv('y_sample.csv').values.flatten()
    feat_names_df = pd.read_csv('features_sample.csv')
    
    # Cache the SHAP explainer to prevent memory crashes
    explainer = shap.TreeExplainer(s2)
    
    return s1, s2, le, X_scaled_data, y_actual_labels, feat_names_df, explainer

# Initialize assets
s1, s2, le, X_test_scaled, y_test, X_orig, cached_explainer = load_all_assets()

# Mapping from notebook
risk_mapping = {
    'Normal':         {'score': 0,  'level': 'None',      'action': 'Allow traffic.'},
    'Generic':        {'score': 2,  'level': 'Very Low',  'action': 'Log and monitor for patterns.'},
    'Fuzzers':        {'score': 4,  'level': 'Low',       'action': 'Update firewall rules to rate-limit source.'},
    'Reconnaissance': {'score': 5,  'level': 'Low',       'action': 'Block source IP; check for open ports.'},
    'DoS':            {'score': 7,  'level': 'High',      'action': 'Activate DDoS mitigation; scrub traffic.'},
    'Exploits':       {'score': 8,  'level': 'High',      'action': 'Isolate affected host; patch vulnerability.'},
    'Shellcode':      {'score': 9,  'level': 'Very High', 'action': 'Immediate quarantine; memory dump analysis.'},
    'Analysis':       {'score': 9,  'level': 'Very High', 'action': 'Deep Packet Inspection; alert human analyst.'},
    'Backdoor':       {'score': 10, 'level': 'Critical',  'action': 'Emergency Shutdown; full system forensic audit.'},
    'Worms':          {'score': 10, 'level': 'Critical',  'action': 'Network Segment Isolation to prevent spread.'}
}

# --- 2. SIDEBAR PANEL ---
st.sidebar.header("🛡️ IDS Control Center")
st.sidebar.info("Select a network packet ID to run a deep forensic audit.")

sample_id = st.sidebar.slider("Select Packet ID", 0, len(X_test_scaled)-1, 0)

# --- 3. PREDICTION LOGIC (Matched to Notebook Pipeline) ---
input_row = X_test_scaled[sample_id].reshape(1, -1)
actual_label = le.classes_[int(y_test[sample_id])]

# Stage 1: Binary Detection
is_attack = s1.predict(input_row)[0]

# UI Layout
st.title("🛡️ Advanced Intrusion Detection System")
st.markdown("---")

if is_attack == 0:
    st.success(f"### ✅ STATUS: CLEAN TRAFFIC")
    st.write(f"The system analyzed this packet and determined it is **Normal**. (Ground Truth: {actual_label})")
    st.metric("Detection Confidence", "100%")
else:
    # Stage 2: Specialist Classification
    attack_classes = [cls for cls in le.classes_ if cls != 'Normal']
    probs = s2.predict_proba(input_row)[0]
    
    # Get indices for specific logic from your notebook
    analysis_s2_idx = attack_classes.index('Analysis')
    backdoor_s2_idx = attack_classes.index('Backdoor')
    worms_s2_idx    = attack_classes.index('Worms')

    # Apply your refined thresholds from notebook
    if probs[analysis_s2_idx] > 0.25:
        pred_idx = analysis_s2_idx
    elif probs[backdoor_s2_idx] > 0.20:
        pred_idx = backdoor_s2_idx
    elif probs[worms_s2_idx] > 0.30:
        pred_idx = worms_s2_idx
    else:
        pred_idx = np.argmax(probs)

    detected_name = attack_classes[pred_idx]
    risk = risk_mapping.get(detected_name, risk_mapping['Normal'])

    # --- 4. DISPLAY METRICS & REPORT ---
    st.error(f"### ⚠️ ALERT: {detected_name.upper()} DETECTED")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk Score", f"{risk['score']}/10", delta_color="inverse")
    m2.metric("Actual Label", actual_label)
    m3.metric("AI Confidence", f"{np.max(probs)*100:.1f}%")

    st.subheader("📋 Security Incident Report")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Threat Level:** {risk['level']}")
    with col_b:
        st.write(f"**Recommended Action:** {risk['action']}")

    # --- 5. EXPLAINABLE AI (SHAP) ---
    st.markdown("---")
    st.subheader("🔍 Forensic Evidence (Explainable AI)")
    st.write("This chart shows which features (packet attributes) forced the AI to flag this as an attack.")

    shap_values = cached_explainer.shap_values(input_row, check_additivity=False)
    
    plt.clf()
    # Handle list-type SHAP values (common for Random Forest)
    if isinstance(shap_values, list):
        sv = shap_values[pred_idx][0]
        base_val = cached_explainer.expected_value[pred_idx]
    else:
        sv = shap_values[0, :, pred_idx]
        base_val = cached_explainer.expected_value[pred_idx]

    shap.force_plot(
        base_val, 
        sv, 
        feature_names=X_orig.columns.tolist(), 
        matplotlib=True, 
        show=False
    )
    st.pyplot(plt.gcf())

st.markdown("---")
st.caption("AI-IDS Portfolio Project | Built with Streamlit & Scikit-Learn")