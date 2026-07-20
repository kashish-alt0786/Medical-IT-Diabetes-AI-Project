import streamlit as st


def show_results(
    t,
    risk_percent,
    risk_level,
    color,
    top_reasons,
    input_df=None
):
    st.header(t["result_header"])

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            "Diabetes Risk",
            f"{risk_percent}%"
        )

        if risk_level == "Low":
            st.success(
                f"**{t['low_risk']} ({risk_percent}%)**\n\n"
                f"{t['low_desc']}"
            )

        elif risk_level == "Moderate":
            st.warning(
                f"**{t['mod_risk']} ({risk_percent}%)**\n\n"
                f"{t['mod_desc']}"
            )

        else:
            st.error(
                f"**{t['high_risk']} ({risk_percent}%)**\n\n"
                f"{t['high_desc']}"
            )

    with col2:

        st.subheader("Main Risk Factors")

        feature_map = {
            "Glucose": "Blood Sugar",
            "BMI": "BMI",
            "Age": "Age",
            "BloodPressure": "Blood Pressure",
            "Family History": "Family History"
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
                f"{icon} **{name}** — Impact: {impact:+.2f}"
            )

    st.divider()

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
