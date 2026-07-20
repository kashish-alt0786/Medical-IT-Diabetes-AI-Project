import streamlit as st

def show_results(t, risk_percent, risk_level, color, top_reasons, input_df):
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Diabetes Risk", f"{risk_percent}%")
        st.markdown(f"#### {t['result_header']}")
        if risk_level == "Low":
            st.success(f"**{t['low_risk']} {risk_percent}%**\n\n{t['low_desc']}")
        elif risk_level == "Moderate":
            st.warning(f"**{t['mod_risk']} {risk_percent}%**\n\n{t['mod_desc']}")
        else:
            st.error(f"**{t['high_risk']} {risk_percent}%**\n\n{t['high_desc']}")

    with col2:
        st.subheader("Main Reasons (XAI)")
        for feat, val in top_reasons[:3]:
            icon = "🔺" if val > 0.2 else "🔹"
            st.write(f"{icon} **{feat}**: impact {val:+.2f}")

    # Personalized Tips
    st.markdown(f"### {t['health_tips']}")
    st.caption(t["tips_desc"])
    if risk_level == "Low":
        st.markdown(t["low_tips_title"])
        st.markdown(t["low_tips"])
    elif risk_level == "Moderate":
        st.markdown(t["mod_tips_title"])
        st.markdown(t["mod_tips"])
    else:
        st.markdown(t["high_tips_title"])
        st.markdown(t["high_tips"])
