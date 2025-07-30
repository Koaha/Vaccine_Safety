
# Sankey Diagram
![Sankey](../plots/sankey_symptom_outcome.html)
## Interpretation
- **Nodes**: Symptoms (left) to outcomes (right).
- **Trends**: Thick flows from 'noi_ban' to 'Full recovery' indicate common mild AEs; 'kho_tho' to 'Ongoing' suggests prolonged issues.
- **Statistical Insight**: Chi-square test for 'kho_tho' vs outcome (p={stats.chi2_contingency(pd.crosstab(df['kho_tho'], df['ket_qua']))[1]:.4f}).
- **Conclusion**: TAK shows low systemic risk; monitor respiratory symptoms for prolonged outcomes.
