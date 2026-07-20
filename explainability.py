import shap
import matplotlib.pyplot as plt


def create_shap_plot(model, input_df, t):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(input_df)

    plt.figure(figsize=(8,5))

    shap.summary_plot(
        shap_values,
        input_df,
        plot_type="bar",
        show=False
    )

    plt.title(t["chart_title"])

    fig = plt.gcf()

    return fig
