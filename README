# Vaccine Safety Analysis Project

## Overview

This project evaluates the safety of the TAK vaccine using a synthetic dataset of Serious Adverse Events (SAEs) from vaccine reports. The analysis follows a rigorous, data-driven approach to clean, explore, model, and interpret SAE data, aiming to identify risk factors, estimate adverse event probabilities, and assess causality. The workflow incorporates best practices in pharmacovigilance and data science as of 2025, leveraging advanced techniques like causal inference, survival analysis, and NLP to provide actionable insights for vaccine safety monitoring.

The project is implemented in Python, using libraries such as `pandas`, `scikit-learn`, `xgboost`, `causalml`, `lifelines`, `sentence-transformers`, and `plotly` for data processing, modeling, and visualization. The analysis produces detailed markdown reports and interactive visualizations to facilitate interpretation by stakeholders, including a Dash dashboard for dynamic exploration.

## Project Objectives

- **Data Quality**: Ensure the dataset is clean, consistent, and reliable for analysis.
- **Exploratory Analysis**: Identify patterns, correlations, and risk factors associated with SAEs.
- **Predictive Modeling**: Develop models to predict severe AEs and identify key features.
- **Causal Inference**: Estimate the causal effect of risk factors (e.g., allergies) on AEs.
- **Temporal Analysis**: Assess AE onset timing and update safety beliefs over time.
- **Reporting**: Generate comprehensive reports and visualizations for actionable insights.

## Project Structure

```
vaccine_sae_analysis/
├── dataset/
│   └── synthetic_vaccine_sae_data.csv  # Synthetic SAE dataset
├── plots/                             # Output directory for visualizations
│   ├── correlation_heatmap.png
│   ├── age_violin.png
│   ├── feature_pairplot.png
│   ├── ae_rates_bar.png
│   ├── onset_boxplot.png
│   ├── age_histogram.html
│   ├── sankey_symptom_outcome.html
│   ├── shap_summary.png
│   ├── bayesian_update.png
│   ├── kaplan_meier.png
│   └── interactive_scatter.html
├── summaries/                         # Output directory for markdown reports
│   ├── descriptive_stats.md
│   ├── association_stats.md
│   ├── correlation_heatmap_interpretation.md
│   ├── age_violin_interpretation.md
│   ├── feature_pairplot_interpretation.md
│   ├── ae_rates_bar_interpretation.md
│   ├── onset_boxplot_interpretation.md
│   ├── age_histogram_interpretation.md
│   ├── sankey_interpretation.md
│   ├── shap_summary_interpretation.md
│   ├── bayesian_update_interpretation.md
│   ├── kaplan_meier_interpretation.md
│   ├── cox_ph_interpretation.md
│   ├── cluster_interpretation.md
│   ├── chi_square_interpretation.md
│   ├── data_quality_profile.md
│   ├── safety_metrics.md
│   └── final_report.md
├── vaccine_sae_analysis_complete.py   # Main analysis script
└── README.md                         # Project documentation
```

## Analysis Plan and Implementation Steps

### Step 1: Data Cleaning and Quality Assurance

#### 1.1 Convert Types and Handle Missing Values
- **What**: Convert columns to appropriate data types (numerical, categorical, datetime) and impute missing values.
- **Why**: Ensures data consistency for downstream analysis; missing values can bias results in pharmacovigilance studies.
- **How**: 
  - Numerical columns (e.g., `age`, `onset_hour`) are coerced to numeric types using `pd.to_numeric`.
  - Categorical columns (e.g., `vung_yk`, `ket_qua`) are filled with 'Unknown' and cast to strings.
  - Date columns (e.g., `vaccine_1_date`) are converted to datetime with `pd.to_datetime` and forward-filled.
  - Text columns (e.g., `mo_ta_dien_bien`) are filled with 'No description'.
  - Rows with all date columns missing are dropped to maintain temporal integrity.
- **Tools**: `pandas`, `numpy`.

#### 1.2 Check Data Consistency
- **What**: Enforce boundaries and formats (e.g., `age` between 0-120, dates in ISO format).
- **Why**: Prevents unrealistic values (e.g., negative ages) that could skew analyses.
- **How**: 
  - Clip numerical values (e.g., `age.clip(0, 120)`).
  - Ensure datetime consistency using `pd.to_datetime`.
- **Tools**: `pandas`.

#### 1.3 Check Data Logic and Conflicts
- **What**: Validate logical relationships (e.g., mutually exclusive genders, valid vaccine timelines).
- **Why**: Logical inconsistencies (e.g., `onset_date` before `vaccine_1_date`) indicate data errors, common in AE datasets like VAERS.
- **How**: 
  - Check gender exclusivity (`female` and `male` cannot both be 1 or 0).
  - Nullify second-dose fields if `so_mui_vaccine` < 2.
  - Drop rows where `onset_date` ≤ `vaccine_1_date`.
  - Reset indices to prevent duplication errors during filtering.
- **Tools**: `pandas`.

#### 1.4 Enhanced Processing
- **What**: Generate data quality metrics (completeness, skewness, kurtosis) and detect outliers.
- **Why**: Provides a comprehensive data profile to assess reliability; outliers can distort statistical models.
- **How**: 
  - Compute completeness (`df.notnull().mean()`), skewness, and kurtosis for numerical columns.
  - Use Isolation Forest (`sklearn.ensemble.IsolationForest`) to flag outliers (5% contamination).
  - Save metrics to `data_quality_profile.md`.
- **Tools**: `pandas`, `scipy`, `sklearn`.

### Step 2: Exploratory Data Analysis (EDA) and Feature Engineering

#### 2.1 Identify and Group Targets
- **What**: Create target variables for modeling (e.g., `has_severe_AE`, `recovery_status`, `ae_severity_score`).
- **Why**: Defines outcomes for predictive and causal analyses; aligns with WHO’s severity grading for AEs.
- **How**: 
  - `has_severe_AE`: Binary indicator (1 if `phan_ve_do_3` or `phan_ve_do_4` is 1).
  - `recovery_status`: Maps `ket_qua` to numerical values (e.g., Full recovery=1, Death=0).
  - `ae_severity_score`: Weighted sum of severity levels (`phan_ve_do_1-4` * [1,2,3,4]).
- **Tools**: `pandas`.

#### 2.2 Feature Engineering
- **What**: Create new features (temporal, comorbidity, encoded categoricals, text embeddings).
- **Why**: Enhances model performance by capturing temporal patterns, risk factors, and unstructured data insights.
- **How**: 
  - Temporal: Compute `time_to_onset` (`(onset_date - vaccine_1_date).days * 24 + onset_hour`) and `ae_duration` (`ket_thuc_time`).
  - Comorbidity: `has_allergy` as max of allergy columns.
  - One-hot encoding: Encode `vung_yk`, `ket_qua`, `vaccine_1_name` using `OneHotEncoder`.
  - NLP: Apply TF-IDF (`TfidfVectorizer`) and Sentence Transformers (`all-MiniLM-L6-v2`) to `mo_ta_dien_bien` for text features.
  - Normalize numerical features with `MinMaxScaler`.
- **Tools**: `pandas`, `sklearn`, `sentence_transformers`.

#### 2.3 Descriptive Statistics and Visualizations
- **What**: Compute AE rates, correlations, and generate visualizations (heatmap, violin, pairplot, bar, boxplot, histogram, Sankey).
- **Why**: Identifies patterns, risk factors, and associations; visualizations aid stakeholder communication per 2025 pharmacovigilance standards.
- **How**: 
  - **Stats**: Calculate AE rate, PRR (vs WHO baseline 0.001), chi-square p-values for categoricals.
  - **Heatmap**: Pearson correlations (`sns.heatmap`) for numerical features.
  - **Violin Plot**: Age distribution by AE status (`sns.violinplot`).
  - **Pairplot**: Pairwise feature relationships (`sns.pairplot`).
  - **Bar Plot**: AE rates by severity score (`sns.barplot`).
  - **Boxplot**: Time to onset by outcome (`sns.boxplot`).
  - **Histogram**: Interactive age distribution (`px.histogram`).
  - **Sankey Plot**: Symptom-to-outcome flows (`go.Sankey`) with expanded symptoms (e.g., `phu_niem`, `sot`).
  - Save plots to `plots/` and interpretations to `summaries/`.
- **Tools**: `pandas`, `seaborn`, `plotly`, `scipy`.

### Step 3: Advanced Modeling

#### 3.1 Predictive Modeling and Feature Selection
- **What**: Train an XGBoost model to predict `has_severe_AE` and select key features.
- **Why**: XGBoost handles non-linear relationships; feature selection identifies risk drivers, per 2025 FDA AI guidelines.
- **How**: 
  - Train XGBoost with Optuna hyperparameter tuning (`max_depth`, `learning_rate`, etc.).
  - Evaluate with AUC and F1-score.
  - Use SHAP (`shap.TreeExplainer`) for feature importance visualization.
  - Apply RFE (`sklearn.feature_selection.RFE`) to select top 10 features.
- **Tools**: `xgboost`, `optuna`, `shap`, `sklearn`.

#### 3.2 Causal Inference Modeling
- **What**: Estimate causal effect of `has_allergy` on `has_severe_AE` using PSM and TMLE.
- **Why**: Observational data is confounded; causal methods like TMLE provide robust effect estimates, aligning with 2025 pharmacovigilance trends.
- **How**: 
  - **PSM**: Fit logistic regression for propensity scores, match treated/control using KDTree, compute ATE.
  - **TMLE**: Use logistic models for outcome (`Q`) and treatment (`g`), apply fluctuation step, estimate ATE.
- **Tools**: `statsmodels`, `scipy`, `causalml`.

#### 3.3 Data Preparation for Time-Based Modeling
- **What**: Sort data by `onset_date` for temporal analysis.
- **Why**: Enables sequential updating of AE probabilities, mimicking real-world monitoring.
- **How**: Sort DataFrame using `df.sort_values('onset_date')`.
- **Tools**: `pandas`.

#### 3.4 Build and Update Bayesian Model
- **What**: Update posterior AE probabilities using a Beta distribution.
- **Why**: Incorporates prior knowledge (WHO baseline) and updates beliefs as new data arrives, per pharmacovigilance standards.
- **How**: 
  - Initialize Beta prior (α=1, β=999 for rare events).
  - Update parameters with observed AEs, compute mean posterior probability.
  - Plot posterior over time (`px.line`).
- **Tools**: `pandas`, `plotly`.

#### 3.5 Other Advanced Techniques
- **What**: Apply survival analysis, clustering, and anomaly detection.
- **Why**: Survival analysis captures AE timing; clustering identifies patient risk profiles; anomalies flag rare events.
- **How**: 
  - **Survival**: Kaplan-Meier curve (`lifelines.KaplanMeierFitter`) and Cox PH model (`CoxPHFitter`) for time-to-AE.
  - **Clustering**: K-means (`sklearn.cluster.KMeans`) with 4 clusters.
  - **Anomaly Detection**: Isolation Forest (`sklearn.ensemble.IsolationForest`).
- **Tools**: `lifelines`, `sklearn`.

### Step 4: Interpretation, Reporting, and Iteration
- **What**: Synthesize results, create a Dash dashboard, and perform sensitivity analysis.
- **Why**: Consolidates findings for stakeholders; sensitivity tests robustness; dashboard enables interactive exploration.
- **How**: 
  - Generate final report with AE rate, PRR, ATEs, and key risk factors (`final_report.md`).
  - Build Dash dashboard with scatter and Sankey plots (`dash.Dash`).
  - Rerun XGBoost without `age` to assess sensitivity.
- **Tools**: `dash`, `plotly`, `xgboost`.

## Prerequisites

- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `sklearn`, `xgboost`, `shap`, `optuna`, `torch`, `lifelines`, `causalml`, `sentence-transformers`, `dash`, `statsmodels`, `scipy`
- Install via: `pip install -r requirements.txt`

## Usage

1. Place `synthetic_vaccine_sae_data.csv` in the `dataset/` directory.
2. Run the main script:
   ```bash
   python vaccine_sae_analysis_complete.py
   ```
3. View outputs in `plots/` and `summaries/`.
4. Access the interactive dashboard at `http://127.0.0.1:8050`.

## Notes

- The dataset is synthetic but designed to mimic real-world SAE reports.
- The analysis assumes a single vaccine (TAK) for simplicity; extend to multiple vaccines by modifying the filtering step.
- Limitations include potential underreporting and lack of control groups, typical in observational AE data.
- Future work could integrate real-time data updates or additional causal methods (e.g., IV regression).

## License

MIT License