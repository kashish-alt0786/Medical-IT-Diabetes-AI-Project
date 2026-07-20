import streamlit as st


def show_results(t, risk_percent, risk_level, top_reasons):

    st.header(t["result_header"])

    col1, col2 = st.columns([1, 2])

    with col1:

        st.metric(
            t["diabetes_risk"],
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

        st.subheader(t["main_reasons"])

        feature_map = {
            "Glucose": t["glucose"],
            "BMI": "BMI",
            "Age": t["age"],
            "BloodPressure": t["bp_status"],
            "Family History": t["family"]
        }

        for feature, impact in top_reasons[:3]:

            display_name = feature_map.get(feature, feature)

            if impact >= 0.35:
                icon = "🔴"
            elif impact >= 0.15:
                icon = "🟠"
            else:
                icon = "🟢"

            st.write(
                f"{icon} **{display_name}** — Impact: {impact:+.2f}"
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
