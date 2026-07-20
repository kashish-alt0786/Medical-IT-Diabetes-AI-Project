import pandas as pd
import numpy as np


def predict_risk(model, feature_names,
                 pregnancies,
                 glucose,
                 bp,
                 skin,
                 insulin,
                 bmi,
                 dpf,
                 age):

    input_data = np.array([[
        pregnancies,
        glucose,
        bp,
        skin,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_df = pd.DataFrame(
        input_data,
        columns=feature_names
    )

    probability = model.predict_proba(input_df)[0][1]

    return probability * 100, input_df
