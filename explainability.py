import matplotlib.pyplot as plt


def create_shap_plot(top_reasons, t):
    """
    Creates a lightweight feature importance chart.

    This replaces SHAP for single-patient explanation and
    avoids compatibility issues with newer XGBoost versions.
    """

    if not top_reasons:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5,
            0.5,
            "No major contributing factors",
            ha="center",
            va="center",
            fontsize=12
        )
        ax.axis("off")
        return fig

    feature_map = {
        "Glucose": "Blood Glucose",
        "BMI": "BMI",
        "Age": "Age",
        "BloodPressure": "Blood Pressure",
        "Family History": "Family History"
    }

    features = [
        feature_map.get(name, name)
        for name, _ in top_reasons
    ]

    values = [score for _, score in top_reasons]

    colors = []

    for value in values:
        if value >= 0.40:
            colors.append("#d62728")     # red
        elif value >= 0.20:
            colors.append("#ff7f0e")     # orange
        else:
            colors.append("#2ca02c")     # green

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.barh(features, values, color=colors)

    ax.set_xlabel("Relative Importance")
    ax.set_title(t.get("chart_title", "Factors Affecting Prediction"))

    ax.invert_yaxis()

    for i, value in enumerate(values):
        ax.text(
            value + 0.01,
            i,
            f"{value:.2f}",
            va="center",
            fontsize=10
        )

    plt.tight_layout()

    return fig
