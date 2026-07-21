import streamlit as st


def show_results(
    t,
    risk_percent,
    risk_level,
    color,
    top_reasons,
    input_df=None,
):

    # ============================================
    # RESULT HEADER
    # ============================================

    st.header("📋 " + t["result_header"])

    st.caption(
        "AI-powered diabetes risk estimation based on your health information."
    )

    st.markdown("---")

    # ============================================
    # MAIN RISK SCORE
    # ============================================

    left, right = st.columns([1, 2])

    with left:

        if risk_level == "Low":
            st.success(f"## {risk_percent}%")

        elif risk_level == "Moderate":
            st.warning(f"## {risk_percent}%")

        else:
            st.error(f"## {risk_percent}%")

        st.metric(
            label=t["diabetes_risk"],
            value=f"{risk_percent}%"
        )

    with right:

        if risk_level == "Low":

            st.success(
                f"""
### 🟢 {t["low_risk"]}

{t["low_desc"]}
"""
            )

        elif risk_level == "Moderate":

            st.warning(
                f"""
### 🟠 {t["mod_risk"]}

{t["mod_desc"]}
"""
            )

        else:

            st.error(
                f"""
### 🔴 {t["high_risk"]}

{t["high_desc"]}
"""
            )

    st.markdown("---")

    # ============================================
    # QUICK SUMMARY CARDS
    # ============================================

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Risk Level",
            risk_level
        )

    with c2:

        if risk_percent < 30:
            label = "Healthy"

        elif risk_percent < 65:
            label = "Needs Monitoring"

        else:
            label = "Medical Check"

        st.metric(
            "Recommendation",
            label
        )

    with c3:

        if risk_percent < 30:
            emoji = "😊"

        elif risk_percent < 65:
            emoji = "🙂"

        else:
            emoji = "⚠️"

        st.metric(
            "Status",
            emoji
        )

    st.markdown("---")

    # ============================================
    # CLINICAL INTERPRETATION
    # ============================================

    st.subheader("🩺 Clinical Interpretation")

    if risk_level == "Low":

        st.info(
            """
Your predicted diabetes risk is currently **low**.

This suggests your health profile is generally
consistent with people who have a lower probability
of Type 2 Diabetes.

Maintaining healthy eating habits, regular exercise,
and periodic health check-ups is recommended.
"""
        )

    elif risk_level == "Moderate":

        st.warning(
            """
Your estimated diabetes risk is **moderate**.

Although this is **not a diagnosis**, improving
lifestyle habits now may significantly reduce your
future diabetes risk.

Regular blood glucose monitoring is recommended.
"""
        )

    else:

        st.error(
            """
Your estimated diabetes risk is **high**.

This result should **NOT** be considered a medical
diagnosis.

However, consultation with a healthcare professional
and laboratory blood glucose testing are strongly
recommended.
"""
        )

    st.markdown("---")

    # ============================================
    # TOP RISK FACTORS
    # ============================================

    st.subheader("🔍 Main Factors Affecting Your Risk")

    st.caption(
        "These factors had the greatest influence on the AI prediction."
    )

    feature_map = {
        "Glucose": "🩸 Blood Glucose",
        "BMI": "⚖️ Body Mass Index",
        "Age": "🎂 Age",
        "BloodPressure": "❤️ Blood Pressure",
        "Family History": "🧬 Family History"
    }

    if len(top_reasons) == 0:

        st.info(
            "No major contributing factors were detected."
        )

    else:

        for feature, impact in top_reasons:

            display_name = feature_map.get(feature, feature)

            if impact >= 0.40:

                st.error(
                    f"**{display_name}**\n\n"
                    f"Very strong influence on prediction "
                    f"({impact:.2f})"
                )

            elif impact >= 0.20:

                st.warning(
                    f"**{display_name}**\n\n"
                    f"Moderate influence on prediction "
                    f"({impact:.2f})"
                )

            else:

                st.success(
                    f"**{display_name}**\n\n"
                    f"Small influence on prediction "
                    f"({impact:.2f})"
                )

    st.markdown("---")

    # ============================================
    # HEALTH SUMMARY
    # ============================================

    st.subheader("📈 Health Summary")

    col1, col2 = st.columns(2)

    with col1:

        glucose = float(input_df["Glucose"].iloc[0])
        bmi = float(input_df["BMI"].iloc[0])

        st.metric(
            "Blood Glucose",
            f"{glucose:.0f} mg/dL"
        )

        st.metric(
            "BMI",
            f"{bmi:.1f}"
        )

    with col2:

        age = int(input_df["Age"].iloc[0])
        bp = float(input_df["BloodPressure"].iloc[0])

        st.metric(
            "Age",
            age
        )

        st.metric(
            "Blood Pressure",
            f"{bp:.0f}"
        )

    st.markdown("---")

    # ============================================
    # RISK CATEGORY EXPLANATION
    # ============================================

    st.subheader("📚 Understanding Your Risk")

    if risk_level == "Low":

        st.success(
            """
### 🟢 Low Risk

Your current health indicators suggest
a relatively low probability of diabetes.

Continue maintaining:

• Healthy diet

• Regular physical activity

• Healthy body weight

• Annual health screening
"""
        )

    elif risk_level == "Moderate":

        st.warning(
            """
### 🟠 Moderate Risk

Some health indicators suggest
an increased likelihood of developing
Type 2 Diabetes in the future.

Lifestyle improvement at this stage
can significantly reduce future risk.
"""
        )

    else:

        st.error(
            """
### 🔴 High Risk

Multiple risk factors are present.

This prediction indicates that
additional laboratory testing
and consultation with a physician
should be considered.

Early intervention greatly improves
long-term health outcomes.
"""
        )

    st.markdown("---")

    # ============================================
    # AI MODEL CONFIDENCE
    # ============================================

    st.subheader("🤖 AI Assessment")

    confidence = max(risk_percent, 100 - risk_percent)

    st.progress(confidence / 100)

    st.write(
        f"Prediction Confidence: **{confidence:.1f}%**"
    )

    st.caption(
        "Confidence reflects how strongly the model supports its prediction. "
        "It does not represent medical certainty."
    )

    st.markdown("---")

    # ============================================
    # PERSONALIZED HEALTH RECOMMENDATIONS
    # ============================================

    st.subheader("💡 Personalized Health Recommendations")

    st.caption(t["tips_desc"])

    if risk_level == "Low":

        st.success(t["low_tips_title"])
        st.markdown(t["low_tips"])

    elif risk_level == "Moderate":

        st.warning(t["mod_tips_title"])
        st.markdown(t["mod_tips"])

    else:

        st.error(t["high_tips_title"])
        st.markdown(t["high_tips"])

    st.info(t["note"])

    st.markdown("---")

    # ============================================
    # PREVENTION CHECKLIST
    # ============================================

    st.subheader("✅ Diabetes Prevention Checklist")

    checklist_col1, checklist_col2 = st.columns(2)

    with checklist_col1:

        st.checkbox("Exercise 30–45 min daily", disabled=True)
        st.checkbox("Eat vegetables every day", disabled=True)
        st.checkbox("Avoid sugary drinks", disabled=True)
        st.checkbox("Maintain healthy BMI", disabled=True)

    with checklist_col2:

        st.checkbox("Sleep 7–8 hours", disabled=True)
        st.checkbox("Drink enough water", disabled=True)
        st.checkbox("Annual blood sugar check", disabled=True)
        st.checkbox("Manage blood pressure", disabled=True)

    st.markdown("---")

    # ============================================
    # WHEN TO SEE A DOCTOR
    # ============================================

    st.subheader("👨‍⚕️ When Should You Consult a Doctor?")

    if risk_level == "High":

        st.error(
            """
### Immediate Recommendation

Please consider scheduling a medical consultation if you experience:

• Frequent urination

• Constant thirst

• Unexplained weight loss

• Blurred vision

• Persistent fatigue

• Slow wound healing

A laboratory blood glucose test
(HbA1c or Fasting Blood Sugar)
is recommended.
"""
        )

    elif risk_level == "Moderate":

        st.warning(
            """
Consider visiting a healthcare provider if:

• Symptoms continue for several weeks

• Diabetes runs in your family

• Blood pressure remains high

• BMI continues increasing

Routine screening every
6–12 months is recommended.
"""
        )

    else:

        st.success(
            """
No urgent medical action is suggested
based on this AI assessment.

Continue maintaining a healthy lifestyle
and undergo routine health screening.
"""
        )

    st.markdown("---")

    # ============================================
    # EDUCATIONAL NOTE
    # ============================================

    with st.expander("📚 About this AI Prediction"):

        st.markdown(
            """
### How does this system work?

This application uses an **XGBoost Machine Learning model**
trained on the well-known **Pima Indian Diabetes Dataset**.

The prediction considers several health indicators:

- Blood Glucose
- Body Mass Index (BMI)
- Age
- Blood Pressure
- Family History
- Pregnancy History

The Explainable AI (XAI) module highlights
which factors contributed most to your prediction.

---

### Important

This prediction **does not diagnose diabetes.**

Only a qualified healthcare professional
can diagnose diabetes using laboratory tests
and clinical evaluation.
"""
        )

    st.markdown("---")

    # ============================================
    # FINAL DISCLAIMER
    # ============================================

    st.caption("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    st.caption(t["footer_disc"])

    st.caption(t["footer_built"])

    st.caption(t["limitations"])

    st.caption("© 2026 Diabetes Risk Predictor • Explainable AI Screening Tool")
