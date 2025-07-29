
# SHAP Summary Plot Report for TAK Vaccine Safety

![SHAP Summary](../plots/shap_summary.png)

## Detailed Interpretation
**X Axis:** SHAP value (impact on prediction; positive for higher severe AE probability).
**Y Axis:** Features ranked by importance.
**Trends Observed:** Dots colored by feature value; red high values pushing right indicate risk factors.
**Conclusions:** Top features like age with positive SHAP for low values conclude younger patients at higher risk for TAK. Observe clustering for nuanced insights.
