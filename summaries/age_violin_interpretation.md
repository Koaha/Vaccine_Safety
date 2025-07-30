
# Age Violin Plot
![Violin Plot](../plots/age_violin.png)
## Interpretation
- **X-axis**: Severe AE status; **Y-axis**: Age distribution.
- **Trends**: Wider violins at younger ages for severe cases.
- **Statistical Insight**: Median age for severe AEs = {df[df['has_severe_AE'] == 1]['age'].median():.2f}.
- **Conclusion**: TAK may pose higher severity risk in younger groups.
