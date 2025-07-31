
# SHAP Summary Plot Report for TAK Vaccine Safety

![SHAP Summary](../plots/shap_summary.png)

## Interpretation: SHAP-Based Feature Importance

This SHAP plot shows how each feature impacts the model’s prediction for **severe adverse events (AE)**.

### Axes:
- **X-axis:** SHAP value (effect on output); values >0 push prediction **toward more severe AE**.
- **Y-axis:** Ranked features by average impact.

### Color Scale:
- **Red:** High feature values  
- **Blue:** Low feature values  
- Example: If `text_emb_383` is red and pushed far right → high values of `text_emb_383` **increase** severe AE risk.

### Observations:
- **Top Feature:** `text_emb_383` — most influential in the model’s prediction logic.
- **Second Feature:** `onset_hour` — also shows meaningful contribution.
- `text_emb_383` and `onset_hour` likely encode **temporal, demographic, or severity-related risk patterns**.

### Conclusion:
This analysis validates that `text_emb_383` and `onset_hour` are **key drivers of severe AE risk** in this cohort.
Model interpretability ensures transparency and helps guide clinical validation.
