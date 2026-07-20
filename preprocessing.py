# preprocessing.py


def calculate_bmi(height_cm, weight_kg):

    bmi = weight_kg / ((height_cm / 100) ** 2)

    if bmi < 25:
        category = "normal"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"

    return round(bmi, 2), category



def map_blood_pressure(bp_option, t):

    mapping = {
        t.get("bp_low", "Low"): 70,
        t.get("bp_normal", "Normal"): 80,
        t.get("bp_high", "High Blood Pressure"): 100,
        t.get("bp_not_sure", "Not Sure"): 85
    }

    return mapping.get(bp_option, 80)
    
def map_family_history(option, t):
    """
    Convert family history selection into Diabetes Pedigree Function value
    """

    mapping = {
        t["family_no"]: 0.15,
        t["family_1"]: 0.50,
        t["family_2"]: 1.20,
        t["family_not_sure"]: 0.30
    }

    return mapping.get(option, 0.30)



def estimate_glucose(symptom_count):
    """
    Estimate glucose level from symptom count
    """

    if symptom_count == 0:
        return 85

    elif symptom_count == 1:
        return 105

    elif symptom_count == 2:
        return 120

    else:
        return 140
