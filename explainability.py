import shap
import matplotlib.pyplot as plt


def create_shap_plot(model, input_df, t):

    explainer = shap.Explainer(
        model.predict,
        input_df
    )

    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(8, 5))

    shap.plots.bar(
        shap_values[0],
        show=False
    )

    plt.title(t["chart_title"])

    return fig
