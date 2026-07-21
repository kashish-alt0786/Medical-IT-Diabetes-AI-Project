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

    # =====================================================
    # PERSONAL INFORMATION
    # =====================================================

    st.header("📝 " + t["health_info"])
    st.caption("Step 1 of 5 • Personal Information")

    with st.container():

        col1, col2 = st.columns([1, 1])

        age = col1.number_input(
            t["age"],
            min_value=1,
            max_value=120,
            value=30,
            help=t["age_help"]
        )

        col2.info(
            "💡 Regular diabetes screening becomes increasingly important after age 35, especially if you have additional risk factors."
        )

    st.markdown("---")

    # =====================================================
    # BLOOD SUGAR
    # =====================================================

    st.subheader("🩸 " + t["blood_sugar_header"])
    st.caption("Step 2 of 5 • Blood Sugar Assessment")

    knows_glucose = st.radio(
        t["have_test"],
        [
            "🧪 " + t["yes_test"],
            "❓ " + t["no_test"]
        ],
        horizontal=True
    )

    if knows_glucose == "🧪 " + t["yes_test"]:

        st.success("Laboratory value selected")

        glucose = st.number_input(
            t["type_fbs"],
            min_value=50,
            max_value=300,
            value=90,
            help=t["fbs_help"]
        )

    else:

        st.info(
            "Answer the symptom questions below to estimate your fasting blood glucose."
        )

        st.markdown(t["no_test_title"])

        thirsty = st.checkbox(
            "💧 " + t["thirsty"]
        )

        tired = st.checkbox(
            "😴 " + t["tired"]
        )

        pee = st.checkbox(
            "🚻 " + t["pee"]
        )

        symptom_count = sum([
            thirsty,
            tired,
            pee
        ])

        glucose = estimate_glucose(symptom_count)

        st.markdown("### Estimated Fasting Blood Glucose")

        if glucose == 85:

            st.success(t["est_85"])

        elif glucose == 105:

            st.warning(t["est_105"])

        elif glucose == 120:

            st.warning(t["est_120"])

        else:

            st.error(t["est_140"])

    with st.expander("📘 " + t["cheat_sheet"]):

        st.markdown(t["cheat_table"])

    st.markdown("---")

    # =====================================================
    # BODY MEASUREMENTS
    # =====================================================
    col3, col4 = st.columns(2)

    height = col3.number_input(
        t["height"],
        min_value=100,
        max_value=250,
        value=165,
        help="Enter your height without shoes."
    )

    weight = col4.number_input(
        t["weight"],
        min_value=30,
        max_value=200,
        value=65,
        help="Enter your current body weight."
    )

    bmi, bmi_category = calculate_bmi(height, weight)

    bmi_label = t[bmi_category]

    st.markdown("### 📊 Body Mass Index")

    c1, c2 = st.columns([1, 2])

    c1.metric("BMI", f"{bmi:.1f}")

    if bmi_category == "normal":
        c2.success(f"✅ {t['normal']}")
    elif bmi_category == "overweight":
        c2.warning(f"⚠️ {bmi_label}")
    else:
        c2.error(f"🔴 {bmi_label}")

    st.markdown("---")

    # =====================================================
    # HEALTH BACKGROUND
    # =====================================================

    st.subheader("❤️ " + t["health_bg"])
    st.caption("Step 4 of 5 • Medical History")

    col5, col6 = st.columns(2)

    bp_options = [
        t["bp_low"],
        t["bp_normal"],
        t["bp_high"],
        t["bp_not_sure"]
    ]

    bp_option = col5.selectbox(
        t["bp_status"],
        bp_options
    )

    bp = map_blood_pressure(
        bp_option,
        t
    )

    pregnancies = col6.number_input(
        t["pregnancies"],
        min_value=0,
        max_value=20,
        value=0,
        help=t["preg_help"]
    )

    st.markdown("---")

    # =====================================================
    # FAMILY HISTORY
    # =====================================================

    st.subheader("👨‍👩‍👧‍👦 " + t["family"])
    st.caption("Step 5 of 5 • Genetic Risk Assessment")

    family_options = [
        t["family_no"],
        t["family_1"],
        t["family_2"],
        t["family_not_sure"]
    ]

    family_history = st.radio(
        t["family"],
        family_options,
        horizontal=True,
        help=t["family_help"]
    )

    dpf = map_family_history(
        family_history,
        t
    )

    st.info(
        "🧬 Family history is one of the strongest known risk factors for Type 2 Diabetes."
    )

    st.markdown("---")

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
