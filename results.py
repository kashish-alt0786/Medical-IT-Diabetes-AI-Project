import streamlit as st

def show_results(risk_percent, t):
    st.markdown("---")
    st.header(t["result_header"])

    if risk_percent < 30:
        st.success(f"**{t['low_risk']} {risk_percent:.1f}%**")
        st.markdown(t["low_desc"])

        with st.expander(t["health_tips"], expanded=True):
            st.markdown(t["tips_desc"])
            st.markdown(t["low_tips_title"])
            st.markdown(t["low_tips"])
            st.caption(t["note"])

    elif risk_percent < 70:
        st.warning(f"**{t['mod_risk']} {risk_percent:.1f}%**")
        st.markdown(t["mod_desc"])

        with st.expander(t["health_tips"], expanded=True):
            st.markdown(t["tips_desc"])
            st.markdown(t["mod_tips_title"])
            st.markdown(t["mod_tips"])
            st.caption(t["note"])

    else:
        st.error(f"**{t['high_risk']} {risk_percent:.1f}%**")
        st.markdown(t["high_desc"])

        with st.expander(t["health_tips"], expanded=True):
            st.markdown(t["tips_desc"])
            st.markdown(t["high_tips_title"])
            st.markdown(t["high_tips"])
            st.caption(t["note"])
