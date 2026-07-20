import shap
import matplotlib.pyplot as plt
import seaborn as sns


def create_shap_plot(model, input_df, t):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    feature_labels = [
        "Pregnancies",
        "Glucose",
        "Blood Pressure",
        "Skin Thickness",
        "Insulin",
        "BMI",
        "Family History",
        "Age"
    ]

    colors = [
        "#d62728" if value > 0 else "#2ca02c"
        for value in shap_values[0]
    ]

    sns.barplot(
        x=shap_values[0],
        y=feature_labels,
        palette=colors,
        ax=ax
    )

    ax.set_title(t["chart_title"], fontsize=14, fontweight="bold")
    ax.set_xlabel(t["chart_xlabel"])
    ax.axvline(x=0, color="black", alpha=0.3)

    return fig
