import streamlit as st

from config import (
    DEFAULT_INSULIN,
    DEFAULT_SKIN_THICKNESS
)

from preprocessing import (
    calculate_bmi,
    map_blood_pressure,
    map_family_history,
    estimate_glucose
)


def show_input_form(t):

    # ======================================================
    # PAGE HEADER
    # ======================================================

    st.header("📝 " + t["health_info"])
    st.caption("Complete the following health assessment")

    progress = st.progress(0)

    # ======================================================
    # STEP 1 — PERSONAL INFORMATION
    # ======================================================

    progress.progress(20)

    st.markdown("## 👤 Personal Information")

    col1, col2 = st.columns([1, 1])

    with col1:

        age = st.number_input(
            t["age"],
            min_value=1,
            max_value=120,
            value=30,
            help=t["age_help"]
        )

    with col2:

        st.info(
            """
Older age slightly increases the risk of Type 2 Diabetes.

Regular screening is recommended after age 35,
especially if you have additional risk factors.
"""
        )

    st.divider()

    # ======================================================
    # STEP 2 — BLOOD GLUCOSE
    # ======================================================

    progress.progress(40)

    st.markdown("## 🩸 Blood Sugar")

    knows_glucose = st.radio(
        t["have_test"],
        [
            "🧪 " + t["yes_test"],
            "❓ " + t["no_test"]
        ],
        horizontal=True
    )
        if knows_glucose == "🧪 " + t["yes_test"]:

        glucose = st.number_input(
            t["type_fbs"],
            min_value=50,
            max_value=300,
            value=90,
            help=t["fbs_help"]
        )

        if glucose < 100:
            st.success("✅ Normal fasting blood sugar")

        elif glucose < 126:
            st.warning("⚠ Prediabetes range")

        else:
            st.error("🔴 Diabetes range")

    else:

        st.info(
            "We'll estimate your fasting glucose using a few common symptoms."
        )

        st.markdown(t["no_test_title"])

        thirsty = st.checkbox("💧 " + t["thirsty"])
        tired = st.checkbox("😴 " + t["tired"])
        pee = st.checkbox("🚻 " + t["pee"])

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

    with st.expander("📖 " + t["cheat_sheet"]):

        st.markdown(t["cheat_table"])

    # ======================================================
    # STEP 3 — BODY MEASUREMENTS
    # ======================================================

    progress.progress(60)

    st.markdown("## 📏 Body Measurements")

    col3, col4 = st.columns(2)

    with col3:

        height = st.number_input(
            t["height"],
            min_value=100,
            max_value=250,
            value=165,
            help="Enter your height in centimeters."
        )

    with col4:

        weight = st.number_input(
            t["weight"],
            min_value=30,
            max_value=200,
            value=65,
            help="Enter your body weight in kilograms."
        )

    bmi, bmi_category = calculate_bmi(height, weight)

    bmi_label = t[bmi_category]

    st.markdown("### 📊 Body Mass Index")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "BMI",
        f"{bmi:.1f}"
    )

    if bmi < 18.5:
        metric2.info("Underweight")
    elif bmi < 25:
        metric2.success(bmi_label)
    elif bmi < 30:
        metric2.warning(bmi_label)
    else:
        metric2.error(bmi_label)

    if bmi < 25:
        metric3.success("🟢 Healthy")
    elif bmi < 30:
        metric3.warning("🟠 Increased Risk")
    else:
        metric3.error("🔴 High Risk")

    st.info(
        "💡 BMI is one of the strongest predictors of Type 2 Diabetes. "
        "Maintaining a healthy BMI can significantly reduce long-term risk."
    )

    st.divider()

    # ======================================================
    # STEP 4 — HEALTH BACKGROUND
    # ======================================================

    progress.progress(80)

    st.markdown("## ❤️ Health Background")

    col5, col6 = st.columns(2)

    bp_options = [
        t["bp_low"],
        t["bp_normal"],
        t["bp_high"],
        t["bp_not_sure"]
    ]

    with col5:

        bp_option = st.selectbox(
            t["bp_status"],
            bp_options
        )

        bp = map_blood_pressure(
            bp_option,
            t
        )

    with col6:

        pregnancies = st.number_input(
            t["pregnancies"],
            min_value=0,
            max_value=20,
            value=0,
            help=t["preg_help"]
        )

    if bp_option == t["bp_high"]:
        st.warning(
            "⚠ High blood pressure often occurs together with diabetes "
            "and increases cardiovascular risk."
        )
    elif bp_option == t["bp_low"]:
        st.info(
            "Low blood pressure is generally not a major diabetes risk factor."
        )
    else:
        st.success("Blood pressure information recorded.")


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
