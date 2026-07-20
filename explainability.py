explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

plt.style.use(...)

fig, ax = plt.subplots(...)

sns.barplot(...)

...
