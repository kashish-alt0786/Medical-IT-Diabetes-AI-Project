# preprocessing.py

def calculate_bmi(height_cm, weight_kg):
    bmi = weight_kg / ((height_cm / 100) ** 2)

    if bmi < 25:
        label = "Normal"
    elif bmi < 30:
        label = "Overweight"
    else:
        label = "Obese"

    return bmi, label


def map_blood_pressure(bp_option):
    mapping = {
        "Low": 70,
        "Normal": 80,
        "High Blood Pressure": 100,
        "Not Sure": 85
    }

    return mapping.get(bp_option, 85)


def map_family_history(option):
    mapping = {
        "No": 0.15,
        "Yes, 1 family member": 0.5,
        "Yes, 2 or more family members": 1.2,
        "Not Sure": 0.3
    }

    return mapping.get(option, 0.3)


def estimate_glucose(symptom_count):
    if symptom_count == 0:
        return 85
    elif symptom_count == 1:
        return 105
    elif symptom_count == 2:
        return 120
    else:
        return 140
