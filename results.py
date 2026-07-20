import streamlit as st


def show_results(
    t,
    risk_percent,
    risk_level,
    color,
    top_reasons,
    input_df=None,
):
    """
    Displays prediction results.
    Compatible with current predictor.py
    """

    st.header(t["result_header"])

    # --------------------------
    # Risk Meter
    # --------------------------

    left, right = st.columns([1, 2])

    with left:

        st.metric(
            label="Risk",
            value=f"{risk_percent:.1f}%"
        )

        st.progress(min(int(risk_percent), 100))

    with right:

        if risk_level == "Low":

            st.success(
                f"### {t['low_risk']}\n\n"
                f"{t['low_desc']}"
            )

        elif risk_level == "Moderate":

            st.warning(
                f"### {t['mod_risk']}\n\n"
                f"{t['mod_desc']}"
            )

        else:

            st.error(
                f"### {t['high_risk']}\n\n"
                f"{t['high_desc']}"
            )

    st.divider()

    # --------------------------
    # Main Reasons
    # --------------------------

    st.subheader("📌 Main Risk Factors")

    feature_map = {
        "Glucose": t.get("blood_sugar_header", "Blood Sugar"),
        "BMI": "BMI",
        "Age": t.get("age", "Age"),
        "BloodPressure": t.get("bp_status", "Blood Pressure"),
        "Family History": t.get("family", "Family History"),
    }

    for feature, impact in top_reasons:

        name = feature_map.get(feature, feature)

        if impact >= 0.35:
            icon = "🔴"

        elif impact >= 0.15:
            icon = "🟠"

        else:
            icon = "🟢"

        st.write(
            f"{icon} **{name}**   "
            f"(Impact: {impact:+.2f})"
        )

    st.divider()

    # --------------------------
    # Personalized Advice
    # --------------------------

    st.subheader(t["health_tips"])

    st.caption(t["tips_desc"])

    if risk_level == "Low":

        st.markdown(f"### {t['low_tips_title']}")
        st.markdown(t["low_tips"])

    elif risk_level == "Moderate":

        st.markdown(f"### {t['mod_tips_title']}")
        st.markdown(t["mod_tips"])

    else:

        st.markdown(f"### {t['high_tips_title']}")
        st.markdown(t["high_tips"])

    st.info(t["note"])

    # --------------------------
    # Recommendation Box
    # --------------------------

    st.divider()

    st.subheader("🩺 Recommendation")

    if risk_level == "Low":

        st.success(
            "Maintain a healthy diet, exercise regularly, "
            "and consider a yearly health check."
        )

    elif risk_level == "Moderate":

        st.warning(
            "Consider improving your lifestyle, reducing sugar intake, "
            "and scheduling a blood sugar test."
        )

    else:

        st.error(
            "Please consult a healthcare professional as soon as possible. "
            "A laboratory blood glucose test is strongly recommended."
        )

    # --------------------------
    # Educational Note
    # --------------------------

    with st.expander("ℹ️ About this prediction"):

        st.write(
            """
This prediction is generated using an Explainable AI model trained
on the Pima Indians Diabetes Dataset.

The result represents **statistical risk**, not a medical diagnosis.

Always consult a qualified healthcare professional before making
medical decisions.
"""
        )
