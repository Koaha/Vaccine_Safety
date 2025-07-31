
# AE Summary and Chi-Square Association Analysis

## Dataset Overview

- **Total Patients:** 89
- **Severe AE Count:** 23
- **Non-Severe AE Count:** 66
- **Overall Severe AE Rate:** 25.84%

## AE Rate by Vaccine Type
```
vaccine_1_name
TAK    0.258427
```

## Chi-Square Association Results (Top 10 Categorical Variables)

| Variable | Chi-square p-value | Significant |
|----------|--------------------|-------------|
| vung_yk | 0.0566 |  |
| female | 0.9501 |  |
| male | 0.9501 |  |
| di_ung_thuoc | 1.0000 |  |
| di_ung_thuoc_muc_do | 0.7277 |  |
| di_ung_thuc_an | 1.0000 |  |
| di_ung_thuc_an_muc_do | 0.6186 |  |
| di_ung_vaccine | 0.5328 |  |
| di_ung_vaccine_muc_do | 0.4820 |  |
| di_ung_khac | 0.3232 |  |

### Interpretation

- Chi-square tests assess statistical dependence between categorical predictors and the binary AE outcome.
- A variable is **significantly associated** when `p < 0.05`.
- ✅ flags features that could influence AE risk.
- Examples:
    - History of **drug/vaccine/food allergies**
    - **Geographic regions** with uneven AE distribution
- Use significant variables for:
    - Feature selection
    - Stratified modeling
    - Risk factor interpretation
