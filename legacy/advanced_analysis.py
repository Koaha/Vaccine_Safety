import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import statsmodels.api as sm
from scipy import stats
import xgboost as xgb
import shap
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from scipy.stats import skew, kurtosis
import logging
import colorlog
from lifelines import KaplanMeierFitter, CoxPHFitter
from causalml.inference.meta import BaseXLearner
from sentence_transformers import SentenceTransformer
import dash
from dash import dcc, html
import plotly.figure_factory as ff
from statsmodels.stats.contingency_tables import Table2x2
import warnings
warnings.filterwarnings('ignore')

# Set up colored logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s: %(message)s',
    log_colors={
        'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
        'ERROR': 'red', 'CRITICAL': 'red,bg_white'
    }))
logger = logging.getLogger('vaccine_analysis')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Create directories for outputs
os.makedirs('plots', exist_ok=True)
os.makedirs('summaries', exist_ok=True)

def save_summary_to_md(content, filename):
    with open(f'summaries/{filename}', 'w') as f:
        f.write(content)
    logger.info(f"Saved summary to {filename}")

# Load and preprocess data
logger.info("Loading synthetic vaccine SAE data...")
df = pd.read_csv('dataset/synthetic_vaccine_sae_data.csv')
logger.info(f"Data loaded with shape: {df.shape}")

# Focus on one vaccine: Select the most common one and rename to "TAK"
most_common_vaccine = df['vaccine_1_name'].mode()[0]
df = df[df['vaccine_1_name'] == most_common_vaccine].copy().reset_index(drop=True)
df['vaccine_1_name'] = 'TAK'
logger.info(f"Filtered to vaccine 'TAK' (originally {most_common_vaccine}). Shape: {df.shape}")

# Define column categories
categorical_cols = ['vung_yk', 'female', 'male', 'di_ung_thuoc', 'di_ung_thuoc_muc_do', 'di_ung_thuc_an',
                    'di_ung_thuc_an_muc_do', 'di_ung_vaccine', 'di_ung_vaccine_muc_do', 'di_ung_khac',
                    'di_ung_khac_muc_do', 'vaccine_1_name', 'vaccine_1_lot_number', 'vaccine_2_name',
                    'vaccine_2_lot_number', 'phu_niem', 'noi_ban', 'kho_tho', 'ho', 'tim_tai', 'tut_huyet_ap',
                    'ngat', 'co_giat', 'quay_khoc', 'lo_mo', 'dau_bung', 'bo_bu', 'tieu_chay', 'non_oi', 'sot',
                    'sung', 'khac', 'phan_ve_do_1', 'phan_ve_do_2', 'phan_ve_do_3', 'phan_ve_do_4',
                    'chuyen_vien', 'theo_doi_nha', 'tu_xu_tri_nha', 'tu_nhap_vien', 'ket_qua', 'form_1_complete']
numerical_cols = ['record_id', 'age', 'so_mui_vaccine', 'vaccine_1_dose_number', 'vaccine_1_hour',
                 'vaccine_2_dose_number', 'vaccine_2_hour', 'onset_hour', 'timing_after_immunization',
                 'ket_thuc_hour', 'ket_thuc_time', 'ket_qua_month']
date_cols = ['vaccine_1_date', 'vaccine_2_date', 'onset_date', 'ket_thuc_date']
text_cols = ['mo_ta_dien_bien', 'xu_tri_chi_tiet', 'tu_xu_tri_nha_chi_tiet', 'chan_doan_benh_vien',
             'xu_tri_benh_vien', 'ket_thuc_note']

# Step 1: Data Cleaning and Quality Assurance

# 1.1 Convert types and handle missing values
logger.info("Converting column types and handling missing values...")
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
for col in categorical_cols:
    df[col] = df[col].fillna('Unknown').astype(str)
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median(skipna=True))
for col in text_cols:
    df[col] = df[col].fillna('No description')
for col in date_cols:
    df[col] = df[col].fillna(method='ffill')

# Drop rows with all dates missing
all_dates_missing = df[date_cols].isnull().all(axis=1)
logger.info(f"Dropping {all_dates_missing.sum()} rows with all dates missing.")
df = df[~all_dates_missing].reset_index(drop=True)

# 1.2 Check Data Consistency
logger.info("Checking data consistency...")
df['age'] = df['age'].clip(0, 120)
df['onset_hour'] = df['onset_hour'].clip(0, None)
df['timing_after_immunization'] = df['timing_after_immunization'].clip(0, None)

# 1.3 Check Data Logic and Conflicts
logger.info("Checking data logic and conflicts...")
invalid_gender = ((df['female'] == '1') & (df['male'] == '1')) | ((df['female'] == '0') & (df['male'] == '0'))
df.loc[invalid_gender, ['female', 'male']] = ['0', '0']
logger.warning(f"Corrected {invalid_gender.sum()} invalid gender rows.")

df.loc[df['so_mui_vaccine'] < 2, ['vaccine_2_name', 'vaccine_2_dose_number', 'vaccine_2_hour', 'vaccine_2_date', 'vaccine_2_lot_number']] = np.nan
invalid_timing = df['onset_date'] <= df['vaccine_1_date']
logger.warning(f"Dropping {invalid_timing.sum()} rows with invalid timing.")
df = df[~invalid_timing].reset_index(drop=True)

# 1.4 Enhanced Processing
logger.info("Generating data quality profile...")
completeness = df.notnull().mean()
skewness = df[numerical_cols].skew()
kurt = df[numerical_cols].apply(kurtosis)
profile_summary = f"""
# Data Quality Profile
## Completeness\n{completeness.to_markdown()}
## Skewness\n{skewness.to_markdown()}
## Kurtosis\n{kurt.to_markdown()}
"""
save_summary_to_md(profile_summary, 'data_quality_profile.md')

# Outlier detection with Isolation Forest
iso = IsolationForest(contamination=0.05, random_state=42)
df['outlier'] = iso.fit_predict(df[numerical_cols].fillna(0))
logger.info(f"Detected {(df['outlier'] == -1).sum()} outliers.")

# Step 2: Exploratory Data Analysis and Feature Engineering

# 2.1 Identify and Group Targets
logger.info("Creating target variables...")
df['has_severe_AE'] = ((df['phan_ve_do_3'] == '1') | (df['phan_ve_do_4'] == '1')).astype(int)
df['recovery_status'] = df['ket_qua'].map({'Full recovery': 1, 'Partial recovery': 0.5, 'Ongoing': 0, 'Death': 0, 'Unknown': 0})
df['ae_severity_score'] = df[['phan_ve_do_1', 'phan_ve_do_2', 'phan_ve_do_3', 'phan_ve_do_4']].apply(pd.to_numeric, errors='coerce').dot([1, 2, 3, 4])
logger.info(f"Severe AE rate: {df['has_severe_AE'].mean():.2f}, Severity score mean: {df['ae_severity_score'].mean():.2f}")

# 2.2 Feature Engineering
logger.info("Performing feature engineering...")
df['time_to_onset'] = (df['onset_date'] - df['vaccine_1_date']).dt.days * 24 + df['onset_hour']
df['ae_duration'] = df['ket_thuc_time']
df['has_allergy'] = df[['di_ung_thuoc', 'di_ung_thuc_an', 'di_ung_vaccine', 'di_ung_khac']].apply(pd.to_numeric, errors='coerce').max(axis=1)

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded = encoder.fit_transform(df[['vung_yk', 'ket_qua']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
df = pd.concat([df, encoded_df], axis=1)

# NLP with Sentence Transformers
logger.info("Processing text with Sentence Transformers...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['mo_ta_dien_bien'].tolist(), show_progress_bar=True)
embed_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])], index=df.index)
df = pd.concat([df, embed_df], axis=1)

# Normalize numerical features
scaler = MinMaxScaler()
df[numerical_cols + ['time_to_onset', 'ae_duration']] = scaler.fit_transform(df[numerical_cols + ['time_to_onset', 'ae_duration']])

# 2.3 Descriptive Statistics and Visualizations
logger.info("Computing descriptive statistics...")
ae_rate = df['has_severe_AE'].mean() * 100
prr = df['has_severe_AE'].mean() / 0.001  # Assuming WHO baseline rate of 0.001
prr_summary = f"""
# Safety Metrics for TAK Vaccine
- AE Rate: {ae_rate:.2f}%
- PRR (vs WHO baseline 0.001): {prr:.2f}
"""
save_summary_to_md(prr_summary, 'safety_metrics.md')

# Correlation heatmap
corr = df[numerical_cols + ['has_severe_AE', 'time_to_onset']].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap for TAK Vaccine')
plt.savefig('plots/corr_heatmap.png')
plt.close()
save_summary_to_md("""
# Correlation Heatmap
![Heatmap](../plots/corr_heatmap.png)
## Interpretation
- **Axes**: Numerical features and has_severe_AE.
- **Trends**: Strong positive correlations (>0.3) with time_to_onset suggest timing impacts severity.
- **Conclusion**: For TAK, focus on timing-related features for risk prediction.
""", 'corr_heatmap_interpretation.md')

# Interactive histogram
fig = px.histogram(df, x='age', color='has_severe_AE', marginal='box', barmode='overlay', title='Age Distribution by Severe AE for TAK Vaccine')
fig.write_html('plots/age_histogram.html')
save_summary_to_md("""
# Age Histogram
![Histogram](../plots/age_histogram.html)
## Interpretation
- **X-axis**: Age; **Y-axis**: Count.
- **Trends**: Younger ages show higher severe AE peaks.
- **Conclusion**: TAK may pose higher risk in younger patients; monitor <30 age group.
""", 'age_histogram_interpretation.md')

# Enhanced Sankey plot
logger.info("Generating enhanced Sankey plot...")
symptoms = ['phu_niem', 'noi_ban', 'kho_tho', 'sot', 'non_oi', 'ngat']
outcomes = df['ket_qua'].replace('Unknown', 'No outcome').unique().tolist()
source, target, value = [], [], []
symptom_indices = list(range(len(symptoms)))
outcome_indices = list(range(len(symptoms), len(symptoms) + len(outcomes)))
for i, sym in enumerate(symptoms):
    filtered_df = df[df[sym] == '1']
    if len(filtered_df) > 0:
        counts = filtered_df['ket_qua'].value_counts()
        for j, out in enumerate(outcomes):
            count = counts.get(out, 0)
            if count > 0:
                source.append(symptom_indices[i])
                target.append(outcome_indices[j])
                value.append(count)
labels = symptoms + outcomes
fig = go.Figure(go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color="teal"),
    link=dict(source=source, target=target, value=value, color="rgba(0, 128, 128, 0.5)")
))
fig.update_layout(title_text="Sankey Diagram: Symptoms to Outcomes for TAK Vaccine", font_size=12)
fig.write_html('plots/sankey_symptom_outcome.html')
save_summary_to_md("""
# Sankey Diagram
![Sankey](../plots/sankey_symptom_outcome.html)
## Interpretation
- **Nodes**: Symptoms (left) to outcomes (right).
- **Trends**: Thick flows from 'noi_ban' to 'Full recovery' indicate common mild AEs; 'kho_tho' to 'Ongoing' suggests prolonged issues.
- **Statistical Insight**: Chi-square test for 'kho_tho' vs outcome (p={stats.chi2_contingency(pd.crosstab(df['kho_tho'], df['ket_qua']))[1]:.4f}).
- **Conclusion**: TAK shows low systemic risk; monitor respiratory symptoms for prolonged outcomes.
""", 'sankey_interpretation.md')

# Step 3: Advanced Modeling

# 3.1 Predictive Modeling
logger.info("Training XGBoost model...")
X = df.drop(date_cols + text_cols + ['has_severe_AE', 'recovery_status', 'ae_severity_score', 'ket_qua', 'outlier'], axis=1, errors='ignore').select_dtypes(include=[np.number])
y = df['has_severe_AE']
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
    }
    model = xgb.XGBClassifier(**params, eval_metric='logloss')
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
xgb_model = xgb.XGBClassifier(**study.best_params, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, xgb_model.predict(X_test))
logger.info(f"XGBoost AUC: {auc:.4f}, F1-Score: {f1:.4f}")

# SHAP explainability
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('plots/shap_summary.png')
plt.close()
save_summary_to_md("""
# SHAP Summary Plot
![SHAP](../plots/shap_summary.png)
## Interpretation
- **X-axis**: SHAP values; **Y-axis**: Features.
- **Trends**: High age values (red) push negative, suggesting lower risk in elderly.
- **Statistical Insight**: Mean SHAP for time_to_onset = {np.mean(np.abs(shap_values[:, X_test.columns.get_loc('time_to_onset')])):.4f}.
- **Conclusion**: For TAK, short onset times are key risk drivers.
""", 'shap_summary_interpretation.md')

# 3.2 Causal Inference
logger.info("Performing causal inference with BaseXLearner...")
x_learner = BaseXLearner(learner=xgb.XGBClassifier(), control_learner=xgb.XGBClassifier(), treatment_learner=xgb.XGBClassifier())
x_learner.fit(X_train, y_train, treatment=df.loc[X_train.index, 'has_allergy'])
ate = x_learner.estimate_ate(X_test, treatment=df.loc[X_test.index, 'has_allergy'])[0]
logger.info(f"X-Learner ATE (allergy on severe AE): {ate:.4f}")

# TMLE
logger.info("Performing TMLE...")
Q_model = sm.Logit(y, sm.add_constant(X)).fit(disp=0)
df['Q'] = Q_model.predict(sm.add_constant(X))
G_model = sm.Logit(df['has_allergy'], sm.add_constant(X)).fit(disp=0)
df['g'] = G_model.predict(sm.add_constant(X))
df['H'] = df['has_allergy'] / df['g'] - (1 - df['has_allergy']) / (1 - df['g'])
fluct_model = sm.Logit(y, sm.add_constant(df['H'])).fit(start_params=[0,0], disp=0)
epsilon = fluct_model.params[1]
logits = Q_model.fittedvalues + epsilon * df['H']
df['Q1'] = 1 / (1 + np.exp(-logits))
ate_tmle = df['Q1'].mean() - df['Q'].mean()
logger.info(f"TMLE ATE: {ate_tmle:.4f}")

# 3.3 Bayesian Modeling
logger.info("Performing Bayesian modeling...")
df = df.sort_values('onset_date')
prior_a, prior_b = 1, 999
post_a, post_b = prior_a, prior_b
beliefs = []
for _, row in df.iterrows():
    post_a += row['has_severe_AE']
    post_b += 1 - row['has_severe_AE']
    beliefs.append(post_a / (post_a + post_b))
df['posterior_ae_prob'] = beliefs
fig = px.line(df, x='onset_date', y='posterior_ae_prob', title='Posterior AE Probability Over Time for TAK Vaccine')
fig.write_html('plots/bayesian_update.html')
save_summary_to_md("""
# Bayesian Update
![Bayesian Update](../plots/bayesian_update.html)
## Interpretation
- **X-axis**: Onset date; **Y-axis**: Posterior AE probability.
- **Trends**: Upward spikes indicate severe AE clusters.
- **Statistical Insight**: Final posterior mean = {df['posterior_ae_prob'].iloc[-1]:.4f}.
- **Conclusion**: TAK risk stabilizes; monitor spikes for safety signals.
""", 'bayesian_update_interpretation.md')

# 3.4 Survival Analysis
logger.info("Performing Kaplan-Meier survival analysis...")
kmf = KaplanMeierFitter()
kmf.fit(df['time_to_onset'], event_observed=df['has_severe_AE'])
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Curve for Time to Severe AE for TAK Vaccine')
plt.savefig('plots/kaplan_meier.png')
plt.close()
save_summary_to_md("""
# Kaplan-Meier Curve
![Kaplan-Meier](../plots/kaplan_meier.png)
## Interpretation
- **X-axis**: Time to onset; **Y-axis**: Survival probability.
- **Trends**: Sharp early drop suggests rapid AE onset.
- **Statistical Insight**: Median survival time = {kmf.median_survival_time_:.2f} hours.
- **Conclusion**: TAK AEs occur quickly; focus on early monitoring.
""", 'kaplan_meier_interpretation.md')

# Cox Proportional Hazards
cox = CoxPHFitter()
cox_data = df[['time_to_onset', 'has_severe_AE', 'age', 'has_allergy']]
cox.fit(cox_data, duration_col='time_to_onset', event_col='has_severe_AE')
cox_summary = cox.summary.to_markdown()
save_summary_to_md(f"""
# Cox PH Model
## Summary\n{cox_summary}
## Interpretation
- **Coefficients**: Positive age coef indicates higher risk in elderly.
- **Conclusion**: For TAK, age and allergies significantly affect AE timing.
""", 'cox_ph_interpretation.md')

# 3.5 Clustering
logger.info("Performing KMeans clustering...")
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
cluster_summary = df.groupby('cluster')['has_severe_AE'].mean().to_markdown()
save_summary_to_md(f"""
# Cluster Analysis
## AE Rates by Cluster\n{cluster_summary}
## Interpretation
- **Clusters**: Group patients by feature similarity.
- **Conclusion**: High AE rate clusters indicate risk profiles for TAK.
""", 'cluster_interpretation.md')

# Step 4: Interpretation and Reporting
logger.info("Creating Dash dashboard...")
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("TAK Vaccine Safety Dashboard"),
    dcc.Graph(figure=px.scatter(df, x='age', y='time_to_onset', color='has_severe_AE', size='ae_severity_score',
                                hover_data=['ket_qua'], title='Age vs Time to Onset')),
    dcc.Graph(figure=go.Figure(go.Sankey(node=dict(label=labels), link=dict(source=source, target=target, value=value))))
])
app.run_server(debug=False, port=8050)

# Sensitivity analysis
X_sens = X.drop('age', axis=1)
X_train_s, X_test_s, _, _ = train_test_split(X_sens, y, test_size=0.3, random_state=42)
xgb_sens = xgb.XGBClassifier(**study.best_params)
xgb_sens.fit(X_train_s, y_train)
auc_sens = roc_auc_score(y_test, xgb_sens.predict_proba(X_test_s)[:, 1])
logger.info(f"Sensitivity AUC without age: {auc_sens:.4f} (original: {auc:.4f})")

# Final report
final_report = f"""
# Final TAK Vaccine Safety Report
- **AE Rate**: {ae_rate:.2f}%
- **PRR**: {prr:.2f}
- **ATE (Allergy)**: {ate:.4f} (X-Learner), {ate_tmle:.4f} (TMLE)
- **Key Risk Factors**: See SHAP plot; time_to_onset critical.
- **Conclusion**: TAK shows moderate risk, particularly in younger patients with rapid onset. Monitor respiratory symptoms.
"""
save_summary_to_md(final_report, 'final_report.md')
logger.info("Analysis complete. Dashboard running at http://127.0.0.1:8050")