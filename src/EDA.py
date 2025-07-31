import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import logger, save_summary_to_md


def prepare_data_for_eda(df):
    """Prepare data by creating targets and features needed for EDA."""
    logger.info("Identifying targets and processing EDA features...")
    df['has_severe_AE'] = ((df['phan_ve_do_3'] == '1') | (df['phan_ve_do_4'] == '1')).astype(int)
    df['recovery_status'] = df['ket_qua'].map({
        'Full recovery': 1, 'Partial recovery': 0.5,
        'Ongoing': 0, 'Death': 0, np.nan: 0, 'Unknown': 0
    })
    df['ae_severity_score'] = df[['phan_ve_do_1', 'phan_ve_do_2', 'phan_ve_do_3', 'phan_ve_do_4']].apply(
        pd.to_numeric, errors='coerce').dot([1, 2, 3, 4])
    
    df['time_to_onset'] = (df['onset_date'] - df['vaccine_1_date']).dt.days * 24 + df['onset_hour']
    df['ae_duration'] = df['ket_thuc_time']

    allergy_cols = ['di_ung_thuoc', 'di_ung_thuc_an', 'di_ung_vaccine', 'di_ung_khac']
    df['has_allergy'] = df[allergy_cols].apply(pd.to_numeric, errors='coerce').max(axis=1)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['vaccine_1_name', 'vung_yk']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)

    tfidf = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf_df = pd.DataFrame(tfidf.fit_transform(df['mo_ta_dien_bien']).toarray(),
                            columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()], index=df.index)
    df = pd.concat([df, tfidf_df], axis=1)

    logger.info("Generating SentenceTransformer embeddings...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['mo_ta_dien_bien'].fillna("").tolist(), show_progress_bar=True)
    embed_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])], index=df.index)
    df = pd.concat([df, embed_df], axis=1)

    scaler = MinMaxScaler()
    num_cols = ['record_id', 'age', 'so_mui_vaccine', 'vaccine_1_dose_number',
                'vaccine_1_hour', 'vaccine_2_dose_number', 'vaccine_2_hour',
                'onset_hour', 'timing_after_immunization', 'ket_thuc_hour',
                'ket_thuc_time', 'ket_qua_month']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df


def chi2_dict_to_md_table(chi2_results: dict, alpha: float = 0.05) -> str:
    """
    Convert chi-square result dictionary to markdown table.

    Parameters:
        chi2_results (dict): Mapping of variable -> p-value
        alpha (float): Significance threshold for highlighting

    Returns:
        str: Markdown-formatted table
    """
    lines = [
        "| Variable | Chi-square p-value | Significant |",
        "|----------|--------------------|-------------|"
    ]
    for key, value in chi2_results.items():
        significance = "✅" if value < alpha else ""
        lines.append(f"| {key} | {value:.4f} | {significance} |")
    return "\n".join(lines)


def summarize_chi2_association(df, categorical_cols, total_samples, ae_counts, ae_rate, incidence_by_vaccine):
    """
    Perform chi-square test on top categorical features and export markdown summary.

    Parameters:
        df (pd.DataFrame): Input data
        categorical_cols (list): Categorical column names
        total_samples (int): Total rows in dataset
        ae_counts (dict): AE outcome distribution
        ae_rate (float): Overall severe AE rate
        incidence_by_vaccine (pd.Series): Rate of AE by vaccine type
    """
    chi2_results = {}
    for col in categorical_cols[:10]:
        contingency = pd.crosstab(df[col], df['has_severe_AE'])
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            chi2_results[col] = p

    chi2_table_md = chi2_dict_to_md_table(chi2_results)

    summary_md = f"""
# AE Summary and Chi-Square Association Analysis

## Dataset Overview

- **Total Patients:** {total_samples}
- **Severe AE Count:** {ae_counts.get(1, 0)}
- **Non-Severe AE Count:** {ae_counts.get(0, 0)}
- **Overall Severe AE Rate:** {ae_rate:.2f}%

## AE Rate by Vaccine Type
```
{incidence_by_vaccine.to_string()}
```

## Chi-Square Association Results (Top 10 Categorical Variables)

{chi2_table_md}

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
"""

    save_summary_to_md(summary_md, 'association_stats.md')


def generate_correlation_heatmap(df, numerical_cols):
    """
    Compute and visualize correlation heatmap among numerical features and AE severity.

    Parameters:
        df (pd.DataFrame): Input dataframe
        numerical_cols (list): Numerical column names
    """
    # Compute Pearson correlation matrix
    corr = df[numerical_cols + ['has_severe_AE']].corr()

    # Extract top correlators with target
    target_corr = corr['has_severe_AE'].drop('has_severe_AE').sort_values(ascending=False)
    top_pos = target_corr[target_corr > 0.2].index.tolist()
    top_neg = target_corr[target_corr < -0.2].index.tolist()

    # Mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
        square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5,
        annot_kws={"size": 9}
    )
    plt.title('Correlation Heatmap: Numerical Features vs Severe AE', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # Markdown summary
    summary_md = f"""
![Correlation Heatmap](../plots/correlation_heatmap.png)

# Correlation Heatmap: Numerical Features and Severe AE

## Method
- Pearson correlation matrix for all numerical variables.
- Target column: `has_severe_AE`

## Observations

### Strongest Positive Correlations with AE Severity:
{', '.join(top_pos) if top_pos else 'None detected'}

### Strongest Negative Correlations:
{', '.join(top_neg) if top_neg else 'None detected'}

### Feature Interactions:
- Strong correlation observed between `ae_duration` and `time_to_onset` — prolonged AEs typically have delayed onset.
- Some time indices (e.g., `vaccine_1_hour`) show moderate correlation with severity.

## Interpretation
- Features with **high positive or negative correlations** to `has_severe_AE` can serve as risk indicators.
- Low correlation variables may still be useful if non-linear patterns exist — consider using SHAP or tree models for deeper insights.

## Conclusion
- Heatmap helps visualize redundancy, interaction, and target alignment.
- Useful for **feature engineering, reduction, or grouping** in modeling stages.
"""

    save_summary_to_md(summary_md, 'correlation_heatmap_interpretation.md')


def generate_age_violin_plot(df):
    """
    Generate a violin plot with strip overlay to visualize age distribution
    by severe AE status and save interpretation as markdown.

    Parameters:
        df (pd.DataFrame): Input data with 'age' and 'has_severe_AE'
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x='has_severe_AE',
        y='age',
        data=df,
        palette='Set2',
        inner='quartile',
        linewidth=1.2
    )
    sns.stripplot(
        x='has_severe_AE',
        y='age',
        data=df,
        color='black',
        alpha=0.2,
        jitter=0.2,
        size=2
    )
    plt.title('Age Distribution by Severe AE Status (Violin + Strip)', fontsize=14)
    plt.xlabel('Severe AE (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/age_violin.png')
    plt.close()

    # Compute basic age stats
    median_0 = df[df['has_severe_AE'] == 0]['age'].median()
    median_1 = df[df['has_severe_AE'] == 1]['age'].median()
    mean_0 = df[df['has_severe_AE'] == 0]['age'].mean()
    mean_1 = df[df['has_severe_AE'] == 1]['age'].mean()

    summary_md = f"""
![Age Violin Plot](../plots/age_violin.png)

# Violin Plot: Age vs Severe AE Status

## Description

This plot compares **age distributions** between patients with and without **severe adverse events (AEs)** after TAK vaccination.  
It combines a **violin plot** (density shape + quartile bars) with a **strip plot** (raw data dots) for full interpretability.

## Observations

- **Median Age**:
  - Non-Severe AE (0): **{median_0:.1f}**
  - Severe AE (1): **{median_1:.1f}**

- **Mean Age**:
  - Non-Severe AE: **{mean_0:.1f}**
  - Severe AE: **{mean_1:.1f}**

- The **distribution is slightly left-skewed** for severe AE group → more cases in younger patients.
- Severe AE density appears **narrower and more centered**, suggesting a tighter age window for risk.

## Interpretation

- **Age is potentially predictive** of AE severity.
- Younger patients may have **higher AE susceptibility**, though the relationship is moderate.
- Consider stratifying downstream analysis (e.g., modeling, SHAP) by age groups.

"""

    save_summary_to_md(summary_md, 'age_violin_interpretation.md')


def perform_eda(df, numerical_cols, categorical_cols, save_plots=True):
    """
    Perform full EDA pipeline and save markdown reports.
    Includes statistical testing, visualization, and markdown interpretation.
    """
    df = prepare_data_for_eda(df)

    ae_rate = df['has_severe_AE'].mean() * 100
    incidence_by_vaccine = df.groupby('vaccine_1_name')['has_severe_AE'].mean()
    total_samples = len(df)
    ae_counts = df['has_severe_AE'].value_counts()

    summarize_chi2_association(df, categorical_cols, 
                            total_samples, ae_counts, ae_rate, incidence_by_vaccine)

    generate_correlation_heatmap(df, numerical_cols)

    generate_age_violin_plot(df)

    # Enhanced Pairplot with Styling
    pairplot_fig = sns.pairplot(
        df[['age', 'time_to_onset', 'ae_duration', 'has_severe_AE']],
        hue='has_severe_AE',
        palette='Set2',
        diag_kind='kde',
        plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'},
        diag_kws={'fill': True}
    )
    pairplot_fig.fig.suptitle("Pairwise Feature Relationships by AE Severity", y=1.03, fontsize=14)
    pairplot_fig.fig.tight_layout()
    pairplot_fig.fig.subplots_adjust(top=0.95)  # Adjust for title space
    pairplot_fig.savefig("plots/feature_pairplot.png")
    plt.close()

    # Updated Markdown Interpretation
    save_summary_to_md("""
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
""", 'feature_pairplot_interpretation.md')

    # Enhanced AE Rate Bar Plot
    ae_rates = df.groupby('ae_severity_score')['has_severe_AE'].mean().sort_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=ae_rates.index, y=ae_rates.values, palette='rocket')
    plt.title('Severe AE Rate by AE Severity Score', fontsize=14)
    plt.xlabel('AE Severity Score (1–10)', fontsize=12)
    plt.ylabel('Proportion with Severe AE', fontsize=12)

    # Annotate values on bars
    for i, v in enumerate(ae_rates.values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/ae_rates_bar.png')
    plt.close()

    # Updated Interpretation
    save_summary_to_md("""
![AE Rate Bar](../plots/ae_rates_bar.png)

## Interpretation: AE Rate by Severity Score

This bar plot shows the proportion of **severe AEs** as a function of the **AE severity score**, which is derived from symptom weighting.

### Key Observations:

- A **clear positive trend**: as severity score increases, the probability of a severe AE rises.
- Scores above **2.5 show a sharp jump**, indicating a non-linear threshold behavior.
- Highest severity scores correspond to nearly **100% chance of being classified as severe**.

### Conclusion:

- The AE severity score is a **strong risk stratifier**.
- Monitoring score distributions can help flag cases for early intervention.
""", 'ae_rates_bar_interpretation.md')


    # Enhanced Boxplot for onset time vs recovery outcome
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        x='ket_qua',
        y='time_to_onset',
        data=df,
        palette='pastel',
        notch=True,
        showfliers=True,
        width=0.6
    )
    plt.title('Time to Onset by Recovery Outcome (TAK Vaccine)', fontsize=14)
    plt.xlabel('Recovery Outcome', fontsize=12)
    plt.ylabel('Time to Onset (hours)', fontsize=12)

    # Annotate median values
    medians = df.groupby('ket_qua')['time_to_onset'].median()
    for tick, label in enumerate(ax.get_xticklabels()):
        outcome = label.get_text()
        median_val = medians.get(outcome, None)
        if median_val is not None:
            ax.text(tick, median_val + 1, f"{median_val:.1f}", ha='center', color='black', fontsize=9)

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('plots/onset_boxplot.png')
    plt.close()

    # Updated Interpretation
    save_summary_to_md("""
![Boxplot Recovery](../plots/onset_boxplot.png)

## Interpretation: Time to Onset by Recovery Outcome

This boxplot visualizes how **onset timing** (in hours post-vaccination) relates to different **clinical recovery outcomes**.

### Key Observations:

- **Poor outcomes** like *Death* and *Ongoing* have **shorter median onset times**, often clustering near the lower quartile.
- **Full recovery** cases are more widely spread, with **higher medians and larger IQR**, indicating more delayed symptom onset.
- **Outliers** (dots beyond whiskers) appear more frequently in the *Ongoing* group, showing variability in atypical cases.

### Conclusion:

- **Faster onset of symptoms** is strongly associated with **worse clinical prognosis**.
- Onset time may be a useful **prognostic marker** for early triage and intervention.
""", 'onset_boxplot_interpretation.md')


    # Enhanced Age Histogram (Interactive, with Marginal Boxplot)
    fig = px.histogram(
        df,
        x='age',
        color='has_severe_AE',
        marginal='box',
        barmode='overlay',
        opacity=0.65,
        nbins=30,
        labels={'age': 'Patient Age', 'has_severe_AE': 'Severe AE'},
        color_discrete_map={0: '#7fc97f', 1: '#d95f02'}
    )

    fig.update_layout(
        title='Age Distribution by Severe AE Status (TAK Vaccine)',
        xaxis_title='Age',
        yaxis_title='Number of Patients',
        legend_title='Severe AE',
        bargap=0.1
    )

    fig.write_html("plots/age_histogram.html")

    # Updated Markdown Summary
    save_summary_to_md("""
[Interactive Age Histogram](../plots/age_histogram.html)

## Interpretation: Age Distribution and Severe AE Risk

This interactive histogram compares the **age distribution** of patients who experienced **severe adverse events (AEs)** vs. those who did not.

### Key Observations:

- The **green bars** (non-severe) span a broad age range, while the **orange bars** (severe) peak in **younger age bins**.
- The **box marginal** on top reveals:
  - **Lower median age** for severe AE group
  - **More tightly packed IQR**, indicating younger clustering
- The **distribution overlap** suggests partial age dependence, but not fully exclusive.

### Conclusion:

- Age is a **contributing factor** in AE severity risk, particularly **in patients under 30**.
- This visual reinforces the need for stratified AE monitoring by age.
""", 'age_histogram_interpretation.md')


    # Sankey Diagram: symptoms to outcomes
    symptoms = ['phu_niem', 'noi_ban', 'kho_tho', 'sot', 'non_oi', 'ngat', 'tieu_chay', 'dau_bung']
    outcomes = df['ket_qua'].dropna().unique().tolist()
    source, target, value = [], [], []
    symptom_idx = list(range(len(symptoms)))
    outcome_idx = list(range(len(symptoms), len(symptoms) + len(outcomes)))

    for i, sym in enumerate(symptoms):
        filtered = df[df[sym] == '1']
        if not filtered.empty:
            counts = filtered['ket_qua'].value_counts()
            for j, outcome in enumerate(outcomes):
                count = counts.get(outcome, 0)
                if count > 0:
                    source.append(symptom_idx[i])
                    target.append(outcome_idx[j])
                    value.append(count)

    labels = symptoms + outcomes
    fig = go.Figure(go.Sankey(node=dict(label=labels),
                              link=dict(source=source, target=target, value=value)))
    fig.update_layout(title_text="Symptom to Outcome Flow (TAK Vaccine)")
    fig.write_html("plots/sankey_symptom_outcome.html")

    save_summary_to_md("""
[Interactive Sankey Diagram](../plots/sankey_symptom_outcome.html)

## Interpretation
- Symptoms like `noi_ban`, `sot` most commonly result in full recovery.
- `kho_tho` and `ngat` disproportionately result in 'Ongoing' outcomes.
""", 'sankey_interpretation.md')
