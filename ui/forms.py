import streamlit as st

from preprocessing import (
    calculate_bmi,
    estimate_glucose,
    map_blood_pressure,
    map_family_history
)

from config import (
    DEFAULT_INSULIN,
    DEFAULT_SKIN_THICKNESS
)

def show_input_form(t):
    pass
# --- User Input ---
st.header(t["health_info"])

col1, col2 = st.columns(2)
age = col1.number_input(t["age"], 1, 120, 30, help=t["age_help"])

# --- Blood Sugar Section ---
st.subheader(t["blood_sugar_header"])
knows_glucose = st.radio(t["have_test"], [t["no_test"], t["yes_test"]], horizontal=True)

if knows_glucose == t["yes_test"]:
    glucose = st.number_input(
        t["type_fbs"],
        50,
        300,
        90,
        help=t["fbs_help"]
    )

else:
    st.markdown(t["no_test_title"])

    thirsty = st.checkbox(t["thirsty"])
    tired = st.checkbox(t["tired"])
    pee = st.checkbox(t["pee"])

    symptom_count = sum([thirsty, tired, pee])
    
    glucose = estimate_glucose(symptom_count)

    if glucose == 85:
        st.success(t["est_85"])
    elif glucose == 105:
        st.warning(t["est_105"])
    elif glucose == 120:
        st.warning(t["est_120"])
    else:
        st.error(t["est_140"])
        
with st.expander(t["cheat_sheet"]):
    st.markdown(t["cheat_table"])

st.subheader(t["body_measure"])
col3, col4 = st.columns(2)
height = col3.number_input(t["height"], 100, 250, 165)
weight = col4.number_input(t["weight"], 30, 200, 65)
bmi, bmi_category = calculate_bmi(height, weight)

bmi_label = t[bmi_category]
st.info(f"{t['bmi_calc']} {bmi:.1f} | {bmi_label}")

st.subheader(t["health_bg"])
col5, col6 = st.columns(2)
bp_options = [t["bp_low"], t["bp_normal"], t["bp_high"], t["bp_not_sure"]]
bp_option = col5.selectbox(t["bp_status"], bp_options)
bp = map_blood_pressure(bp_option, t)

pregnancies = col6.number_input(t["pregnancies"], 0, 20, 0, help=t["preg_help"])

family_options = [t["family_no"], t["family_1"], t["family_2"], t["family_not_sure"]]
family_history = st.radio(t["family"], family_options, horizontal=True, help=t["family_help"])

dpf = map_family_history(family_history, t)

insulin = DEFAULT_INSULIN
skin = DEFAULT_SKIN_THICKNESS

st.divider()

return (
        age,
        glucose,
        bmi,
        bp,
        pregnancies,
        dpf,
        insulin,
        skin
    )
