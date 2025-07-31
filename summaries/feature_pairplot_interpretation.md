
![Feature Pairplot](../plots/feature_pairplot.png)

## Interpretation of Pairwise Feature Plot

This plot displays the relationships between:
- `age`
- `time_to_onset`
- `ae_duration`

across the two outcome groups of severe AE (1) and non-severe (0).

### Key Observations:

- **`time_to_onset` vs `ae_duration`**:
  - Severe AE cases cluster in regions with **short onset time and short duration**.
  - Non-severe AEs are more spread out across longer durations.

- **`age` vs other features**:
  - No strong linear pattern, but **interactions with timing suggest clusters**.
  - Younger age combined with fast onset aligns with higher AE risk.

- **Diagonal KDEs**:
  - KDEs show **more peaked distribution for severe cases**, especially in `time_to_onset`.

### Conclusion:

- The pairwise feature view reveals **potential interaction effects** in AE progression.
- Modeling should incorporate these relationships to better predict severe outcomes.
