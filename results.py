import joblib
import streamlit as st

from config import FEATURE_NAMES


def _render_health_recommendations(text):
    """Render diet guidance followed by daily activity guidance."""
    lifestyle_marker = "**🏃 LIFESTYLE:**"
    if lifestyle_marker in text:
        diet_text, lifestyle_text = text.split(lifestyle_marker, 1)
        st.markdown(diet_text.rstrip())
        st.markdown(lifestyle_marker + lifestyle_text)
    else:
        st.markdown(text)


def _render_feature_sensitivity(input_df, current_risk):
    """Audit model sensitivity by changing selected inputs hypothetically."""
    if input_df is None or input_df.empty:
        return

    st.markdown("---")
    st.subheader("🔬 Interactive Feature Sensitivity Analysis")
    st.caption(
        "Audit how the trained model responds to hypothetical changes in selected input variables. "
        "The original prediction remains unchanged and this simulation is not a real-world health forecast."
    )
    st.info(
        "🧪 **Model Sensitivity Audit:** This panel changes input values and re-runs the same serialized model "
        "to show how its mathematical decision boundary responds. It is a machine-learning validation exercise, "
        "not a prediction of how changing a measurement will change a person's actual health."
    )

    original_glucose = float(input_df["Glucose"].iloc[0])
    original_bmi = float(input_df["BMI"].iloc[0])
    c1, c2 = st.columns(2)
    with c1:
        scenario_glucose = st.slider("Simulated Glucose (mg/dL)", 40.0, 300.0, original_glucose, 1.0, key="sensitivity_glucose")
    with c2:
        scenario_bmi = st.slider("Simulated BMI", 10.0, 70.0, original_bmi, 0.1, key="sensitivity_bmi")

    try:
        model = joblib.load("model.pkl")
        scenario = input_df.copy()
        scenario.loc[scenario.index[0], "Glucose"] = scenario_glucose
        scenario.loc[scenario.index[0], "BMI"] = scenario_bmi
        scenario = scenario.reindex(columns=FEATURE_NAMES)
        scenario_risk = float(model.predict_proba(scenario)[0][1]) * 100
        delta = scenario_risk - float(current_risk)

        a, b, c = st.columns(3)
        a.metric("Baseline Model Output", f"{current_risk:.1f}%")
        b.metric("Sensitivity Scenario", f"{scenario_risk:.1f}%")
        c.metric("Model Output Change", f"{delta:+.1f} percentage points")

        if delta < 0:
            st.success("🟢 The model output moved lower under this hypothetical input configuration.")
        elif delta > 0:
            st.warning("🟠 The model output moved higher under this hypothetical input configuration.")
        else:
            st.info("The selected input changes produced no meaningful change in the model output.")

        st.caption("Interpretation: this demonstrates model sensitivity to the selected features. It does not establish causality, treatment effectiveness, or a change in real-world diabetes risk.")
    except Exception as exc:
        st.info(f"Feature sensitivity analysis is temporarily unavailable: {exc}")


def show_results(t, risk_percent, risk_level, top_reasons, input_df=None):
    st.header("📋 " + t["result_header"])
    st.caption("AI-powered diabetes risk estimation based on your health information.")
    st.markdown("---")

    left, right = st.columns([1, 2])
    with left:
        if risk_level == "Low": st.success(f"## {risk_percent}%")
        elif risk_level == "Moderate": st.warning(f"## {risk_percent}%")
        else: st.error(f"## {risk_percent}%")
        st.metric(label=t.get("diabetes_risk", "Diabetes Risk"), value=f"{risk_percent}%")

    with right:
        if risk_level == "Low": st.success(f"### 🟢 {t.get('low_risk', 'Low Risk')}\n\n{t.get('low_desc', '')}")
        elif risk_level == "Moderate": st.warning(f"### 🟠 {t.get('mod_risk', 'Moderate Risk')}\n\n{t.get('mod_desc', '')}")
        else: st.error(f"### 🔴 {t.get('high_risk', 'High Risk')}\n\n{t.get('high_desc', '')}")

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk Level", risk_level)
    c2.metric("Recommendation", "Healthy" if risk_percent < 30 else "Needs Monitoring" if risk_percent < 65 else "Medical Check")
    c3.metric("Status", "😊" if risk_percent < 30 else "🙂" if risk_percent < 65 else "⚠️")
    st.markdown("---")

    st.subheader("🩺 Clinical Interpretation")
    if risk_level == "Low": st.info("Your predicted diabetes risk is currently **low**. Maintaining healthy eating habits, regular exercise, and periodic health check-ups is recommended.")
    elif risk_level == "Moderate": st.warning("Your estimated diabetes risk is **moderate**. Although this is **not a diagnosis**, improving lifestyle habits now may reduce future risk. Regular health monitoring is recommended.")
    else: st.error("Your estimated diabetes risk is **high**. This result is **not a diagnosis**. Consultation with a healthcare professional and appropriate laboratory testing are recommended.")

    st.markdown("---")
    st.subheader("🔍 Main Factors Affecting Your Risk")
    feature_map = {"Glucose": "🩸 Blood Glucose", "BMI": "⚖️ Body Mass Index", "Age": "🎂 Age", "BloodPressure": "❤️ Blood Pressure", "Family History": "🧬 Family History"}
    for feature, impact in top_reasons:
        name = feature_map.get(feature, feature)
        if impact >= 0.40: st.error(f"**{name}** — Very strong influence ({impact:.2f})")
        elif impact >= 0.20: st.warning(f"**{name}** — Moderate influence ({impact:.2f})")
        else: st.success(f"**{name}** — Small influence ({impact:.2f})")

    st.markdown("---")
    st.subheader("📈 Health Summary")
    if input_df is not None and not input_df.empty:
        a, b = st.columns(2)
        with a:
            st.metric("Blood Glucose", f"{float(input_df['Glucose'].iloc[0]):.0f} mg/dL")
            st.metric("BMI", f"{float(input_df['BMI'].iloc[0]):.1f}")
        with b:
            st.metric("Age", int(input_df['Age'].iloc[0]))
            st.metric("Blood Pressure", f"{float(input_df['BloodPressure'].iloc[0]):.0f}")

    _render_feature_sensitivity(input_df, risk_percent)

    st.markdown("---")
    st.subheader("📚 Understanding Your Risk")
    if risk_level == "Low": st.success("### 🟢 Low Risk\n\nContinue healthy eating, regular physical activity, healthy weight management, and routine screening.")
    elif risk_level == "Moderate": st.warning("### 🟠 Moderate Risk\n\nSome indicators suggest increased likelihood of developing Type 2 Diabetes. Lifestyle improvement and regular screening are important.")
    else: st.error("### 🔴 High Risk\n\nMultiple risk factors are present. Consider additional laboratory testing and consultation with a healthcare professional.")

    st.markdown("---")
    st.subheader("🤖 AI Assessment")
    confidence = abs(risk_percent - 50) * 2
    st.progress(confidence / 100)
    st.write(f"Prediction Confidence: **{confidence:.1f}%**")
    st.caption("Confidence reflects model output strength, not medical certainty.")

    st.markdown("---")
    st.subheader("💡 Personalized Health Recommendations")
    st.caption(t["tips_desc"])
    if risk_level == "Low": st.success(t["low_tips_title"]); _render_health_recommendations(t["low_tips"])
    elif risk_level == "Moderate": st.warning(t["mod_tips_title"]); _render_health_recommendations(t["mod_tips"])
    else: st.error(t["high_tips_title"]); _render_health_recommendations(t["high_tips"])

    st.markdown("---")
    st.subheader("🥗 Manage Your Nutrition")
    st.info("Continue to NutriGuard AI and use this diabetes-risk result to personalize your educational nutrition guidance.")
    nutriguard_url = f"https://nutriguard-ai-rrzi6rnezvcba9dhtgzlrm.streamlit.app/?risk={float(risk_percent):.1f}&source=diabetes-risk-predictor"
    st.link_button("🥗 Continue to NutriGuard AI →", nutriguard_url)

    st.markdown("---")
    st.subheader("✅ Diabetes Prevention Checklist")
    a, b = st.columns(2)
    with a:
        st.checkbox("Exercise 30–45 min daily", disabled=True); st.checkbox("Eat vegetables every day", disabled=True); st.checkbox("Avoid sugary drinks", disabled=True); st.checkbox("Maintain healthy BMI", disabled=True)
    with b:
        st.checkbox("Sleep 7–8 hours", disabled=True); st.checkbox("Drink enough water", disabled=True); st.checkbox("Annual blood sugar check", disabled=True); st.checkbox("Manage blood pressure", disabled=True)

    st.info(t["note"])
    st.markdown("---")
    st.subheader("👨‍⚕️ When Should You Consult a Doctor?")
    if risk_level == "High": st.error("Consider scheduling a medical consultation if you have concerning symptoms or risk factors.")
    elif risk_level == "Moderate": st.warning("Consider visiting a healthcare provider if symptoms continue or diabetes risk factors persist.")
    else: st.success("Continue healthy habits and routine screening.")

    with st.expander("📚 About this AI Prediction"):
        st.markdown("This application uses a machine-learning model trained on the **Pima Indian Diabetes Dataset**. The Explainable AI module highlights factors associated with the prediction. This prediction does **not** diagnose diabetes; clinical diagnosis requires professional evaluation and appropriate laboratory testing.")

    st.markdown("---")
    st.caption(t["footer_disc"])
    st.caption(t["footer_built"])
    st.caption(t["limitations"])
