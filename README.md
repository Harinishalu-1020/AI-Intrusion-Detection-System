# AI-Intrusion-Detection-System
1. Executive Summary
In cybersecurity, "Black Box" AI models create a trust gap. When an automated system flags a network packet, security analysts often lack the forensic evidence needed to verify the threat. This project bridges that gap by combining a high-performance Two-Stage Machine Learning Pipeline with SHAP (SHapley Additive exPlanations). The result is a system that not only detects 9 unique attack types with high precision but also provides a visual "proof" of why each alert was triggered.

# 2. Technical Architecture
The system is designed to be computationally efficient by using a hierarchical classification approach:

# Stage 1: Binary Filtering (Ensemble Logic)
The first layer acts as a high-speed gatekeeper. It is trained to distinguish between Normal traffic and Anomalies. By filtering out legitimate traffic first, the system saves processing power, ensuring that only suspicious packets are sent for deep analysis.

# Stage 2: Specialist Classification (Random Forest)
Once a packet is flagged as an anomaly, it enters Stage 2. This model is a Random Forest classifier trained to identify specific attack categories such as DoS, Fuzzers, Exploits, Backdoors, and Reconnaissance.

# 3. Probability Thresholding & Class Imbalance
To handle high-risk but rare attacks (like Analysis and Backdoors), I implemented Custom Probability Thresholding.

Standard models often miss rare attacks because they "hide" in the data.

I adjusted the logic to trigger alerts even at lower confidence scores (e.g., 20-25%) for these critical threats, ensuring a much higher Recall for the most dangerous activities.

# 4. Explainable AI (XAI) for Forensics
This project utilizes SHAP to provide transparency. For every detection, the system generates a Forensic Force Plot.

Red Bars: Show features that pushed the model to predict an attack (e.g., high source-to-destination bytes).

Blue Bars: Show features that suggested the traffic might be normal.
This allows a human analyst to see the "smoking gun" features (like sttl, sbytes, or dur) instantly, reducing the Mean Time to Respond (MTTR).

5. Interactive SOC Dashboard
The Streamlit dashboard serves as a professional interface for a Security Operations Center (SOC). It includes:

Automated Risk Scoring: A 1-10 severity scale mapped to attack impact.

Tactical Recommendations: Suggested actions (e.g., "Deep Packet Inspection" or "Isolate Segment") based on the specific threat type detected.

Audit Capability: A slider-based investigation panel that allows users to audit historical network packets and view the AI’s decision-making process.

# 6. Impact & Business Value
Reduces Alert Fatigue: Analysts only focus on high-confidence, high-risk alerts.

Verified AI: Moves from "Black Box" predictions to "Glass Box" forensics.

Speed: The two-stage design ensures real-time performance on high-volume network links.
