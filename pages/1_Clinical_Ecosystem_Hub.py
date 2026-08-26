"""Unified clinical-health ecosystem navigation hub."""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Clinical Health Ecosystem", page_icon="🏥", layout="wide")

st.title("🏥 Clinical Health Ecosystem")
st.caption("A unified educational workspace connecting diabetes-risk screening, nutrition guidance and MLOps validation.")

st.markdown(
    """
<div class="clinical-notice">
<b>Institutional Notice:</b> This platform is an educational research framework for chronic-disease prediction and preventative analytics. It is not a certified diagnostic instrument and does not replace licensed medical oversight.
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Health Intelligence Architecture")
tab1, tab2, tab3 = st.tabs([
    "🩺 Diagnostic Screener",
    "🥗 Nutritional Therapeutics",
    "⚙️ MLOps & System Metrics",
])

with tab1:
    st.subheader("Diabetes Risk Predictor")
    st.write("Enter clinical screening inputs and review the statistical risk estimate with explainability support.")
    st.page_link("app.py", label="Open Diagnostic Screener", icon="🩺")

with tab2:
    st.subheader("NutriGuard AI")
    st.write("Continue from risk screening to educational meal and nutrition analysis in the connected NutriGuard application.")
    st.link_button(
        "Open NutriGuard AI",
        "https://nutriguard-ai-rrzi6rnezvcba9dhtgzlrm.streamlit.app/",
        use_container_width=True,
    )
    st.caption("The companion app provides educational nutrition guidance only and is not medical treatment.")

with tab3:
    st.subheader("MLOps & System Validation")
    st.write("Compare Logistic Regression, Random Forest and XGBoost; inspect validation artifacts, training history, drift notes and fairness limitations.")
    st.page_link("pages/3_Model_Metrics_Validation.py", label="Open MLOps Dashboard", icon="📊")

st.markdown("---")
st.markdown(
    "**Data Privacy Accord:** This workspace is designed to operate statelessly. Biometric inputs, uploaded files and prediction outcomes are not written to a persistent patient database by this project."
)
