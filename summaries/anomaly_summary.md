
![Anomaly Detection](../plots/anomaly_detection.png)

# Anomaly Detection Summary

## Method
- Isolation Forest (unsupervised)
- Contamination rate: **5%**
- Projection: PCA (2D)

## Results
- Total detected anomalies: **5**
- Red 'X' marks indicate patients with **unexpected feature profiles**
- PCA captures 100.00% of total variance

## Interpretation
- Isolation Forest separates sparse or extreme data points.
- These flagged records may correspond to:
  - AE outliers
  - Misclassified metadata
  - Rare comorbidity combinations

## Actionable Use
- Prioritize flagged cases for manual validation or investigation
- Could uncover edge-case signals missed by population averages
