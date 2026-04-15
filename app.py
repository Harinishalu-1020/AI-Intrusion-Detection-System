import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# --- LOAD ASSETS (Optimized to prevent crashes) ---
@st.cache_resource
def load_assets():
    s1 = joblib.load('stage1_binary_model.pkl')
    s2 = joblib.load('stage2_specialist_model.pkl')
    le = joblib.load('label_encoder.pkl')
    # Use the pre-scaled data you exported
    X_sample = pd.read_csv('X_sample.csv').values 
    y_sample = pd.read_csv('y_sample.csv').values.flatten()
    feat_names = pd.read_csv('features_sample.csv')
    
    # Create the SHAP explainer ONCE and cache it to prevent memory crashes
    explainer = shap.TreeExplainer(s2)
    return s1, s2, le, X_sample, y_sample, feat_names, explainer

s1, s2, le, X_test_scaled, y_test, X_orig, cached_explainer = load_assets()

# ... (keep your risk_mapping here) ...

# --- GUI ---
st.sidebar.header("Investigation Panel")
sample_id = st.sidebar.slider("Select Packet ID to Audit", 0, len(X_test_scaled)-1, 0)

input_row = X_test_scaled[sample_id].reshape(1, -1)
actual_label = le.classes_[int(y_test[sample_id])]

# Stage 1 Prediction
is_attack = s1.predict(input_row)[0]

if is_attack == 0:
    st.success(f"### STATUS: CLEAN TRAFFIC (Actual: {actual_label})")
else:
    probs = s2.predict_proba(input_row)
    # Your specific logic for Analysis/Backdoor
    attack_classes = [cls for cls in le.classes_ if cls != 'Normal']
    # Use index safely
    try:
        analysis_idx = attack_classes.index('Analysis')
        backdoor_idx = attack_classes.index('Backdoor')
        
        if probs[0][analysis_idx] > 0.25: pred_idx = analysis_idx
        elif probs[0][backdoor_idx] > 0.20: pred_idx = backdoor_idx
        else: pred_idx = np.argmax(probs)
    except:
        pred_idx = np.argmax(probs)

    detected_name = attack_classes[pred_idx]
    
    # ... (Show metrics as you had them) ...

    # --- SHAP Section (Fixed for speed) ---
    st.subheader("🔍 Forensic Evidence (Explainable AI)")
    # Use the cached explainer instead of creating a new one!
    shap_values = cached_explainer.shap_values(input_row, check_additivity=False)
    
    plt.clf()
    if isinstance(shap_values, list):
        sv = shap_values[pred_idx][0]
        base_val = cached_explainer.expected_value[pred_idx]
    else:
        sv = shap_values[0, :, pred_idx]
        base_val = cached_explainer.expected_value[pred_idx]
        
    shap.force_plot(base_val, sv, feature_names=X_orig.columns.tolist(), matplotlib=True, show=False)
    st.pyplot(plt.gcf())