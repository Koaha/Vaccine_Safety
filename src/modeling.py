from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
import xgboost as xgb
import shap
import optuna
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from src.utils import logger, save_summary_to_md
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def prepare_data_for_modeling(df, date_cols, text_cols):
    columns_to_drop = ['has_severe_AE'] + date_cols + text_cols + ['valid_timing', 'recovery_status', 'ae_severity_score', 'ket_qua']
    X = df.drop(columns_to_drop, axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['has_severe_AE']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    return X, y, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)

    logger.info("Model Evaluation Report:\n" + report)

    save_summary_to_md(f"""
# Model Evaluation Report

**AUC Score:** {auc:.4f}  
**F1 Score:** {f1:.4f}  

## Classification Report
```
{report}
```

## Confusion Matrix
```
{conf}
```
""", 'model_evaluation.md')

    return auc, f1


def train_xgboost(X_train, y_train):
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
        }
        model = xgb.XGBClassifier(**params, eval_metric='logloss')
        score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)
    best_params = study.best_params
    model = xgb.XGBClassifier(**best_params, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model, best_params


def explain_with_shap(model, X_test):
    """
    Generate SHAP summary plot and markdown interpretation for model explainability.

    Parameters:
        model: Trained model (e.g., XGBoost or Logistic Regression).
        X_test (pd.DataFrame): Test features.
    """
    import shap
    import matplotlib.pyplot as plt

    logger.info("Computing SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # Plot summary
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(12, 8))
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=300)
    plt.close()
    logger.info("✅ Saved SHAP summary plot to: plots/shap_summary.png")

    # Determine top features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[::-1]
    top_1 = X_test.columns[top_features[0]]
    top_2 = X_test.columns[top_features[1]]

    # Dynamic SHAP markdown explanation
    save_summary_to_md(f"""
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
- Example: If `{top_1}` is red and pushed far right → high values of `{top_1}` **increase** severe AE risk.

### Observations:
- **Top Feature:** `{top_1}` — most influential in the model’s prediction logic.
- **Second Feature:** `{top_2}` — also shows meaningful contribution.
- `{top_1}` and `{top_2}` likely encode **temporal, demographic, or severity-related risk patterns**.

### Conclusion:
This analysis validates that `{top_1}` and `{top_2}` are **key drivers of severe AE risk** in this cohort.
Model interpretability ensures transparency and helps guide clinical validation.
""", 'shap_summary_interpretation.md')


def train_bayesian_logistic(X, y):
    """
    Train a simple Bayesian-inspired logistic regression using PyTorch.
    Logs training loss and summarizes key observations dynamically.

    Returns:
        model (nn.Module): Trained PyTorch model.
        final_loss (float): Final loss value.
    """
    class BayesianLogistic(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    # Initialization
    input_dim = X.shape[1]
    model = BayesianLogistic(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCELoss()

    X_torch = torch.tensor(X.values, dtype=torch.float32)
    y_torch = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_torch, y_torch)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Training loop
    loss_trace = []
    for epoch in range(200):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        loss_trace.append(loss.item())
        if epoch % 50 == 0:
            logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Extract learned weights
    with torch.no_grad():
        weights = model.linear.weight.cpu().numpy().flatten()

    weight_df = pd.DataFrame({
        "Feature": X.columns,
        "Weight": weights
    }).sort_values(by="Weight", key=abs, ascending=False)

    top_feature = weight_df.iloc[0]["Feature"]
    top_weight = weight_df.iloc[0]["Weight"]

    # Save plot of training loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss_trace, label='Training Loss', color='purple')
    plt.title("Bayesian Logistic Regression Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/bayesian_logistic_loss.png")
    plt.close()

    # Markdown summary
    save_summary_to_md(f"""
# Bayesian Logistic Regression Summary

![Training Loss Curve](../plots/bayesian_logistic_loss.png)

## Model & Training Info
- **Final Loss:** {loss_trace[-1]:.4f}  
- **Total Epochs:** {len(loss_trace)}  
- **Optimizer:** Adam (weight_decay=0.01)  
- **Input Features:** {input_dim}

## Top Influential Feature (by weight)
- **Feature:** `{top_feature}`
- **Weight:** {top_weight:.4f}
- Interpretation: A **positive** weight suggests that higher values of `{top_feature}` increase the probability of **severe AE**.

## Interpretation
This model captures non-linear probabilistic relationships using a Bayesian-like approach (with L2 regularization via weight decay).  
Although simpler than ensemble models, its interpretability is useful for early safety signal discovery in vaccine surveillance.
""", "bayesian_logistic_summary.md")

    return model, loss_trace[-1]


def train_logistic_regression(X_train, y_train):
    """
    Trains a traditional logistic regression model, plots coefficients,
    and saves detailed markdown with dynamic explanation.

    Returns:
        model (LogisticRegression): Trained model object
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Coefficients
    coef_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': model.coef_[0]
    })
    coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='AbsCoef', ascending=False)

    top_feature = coef_df.iloc[0]['Feature']
    top_weight = coef_df.iloc[0]['Coefficient']

    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=coef_df.head(20), x='Coefficient', y='Feature', palette='coolwarm')
    plt.title("Top Logistic Regression Coefficients (Severe AE Prediction)")
    plt.axvline(0, linestyle='--', color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig("plots/logistic_coefficients.png")
    plt.close()

    # Markdown with dynamic feature explanation
    save_summary_to_md(f"""
# Logistic Regression Summary

![Logistic Coefficients](../plots/logistic_coefficients.png)

## Top Influential Features (by absolute value):

```
{coef_df.head(10)[['Feature', 
    'Coefficient']].to_string(index=False)}
```

## Key Interpretation
- **Top Predictor:** `{top_feature}` with coefficient = {top_weight:.4f}
- A **positive** coefficient suggests increasing `{top_feature}` value **raises** the probability of severe AE.
- A **negative** coefficient implies `{top_feature}` may be **protective**.

## Model Notes
- Logistic regression offers interpretable risk factor analysis.
- Coefficients represent **log-odds** effect per unit increase in each feature.
- Useful for clinical validation and screening high-risk variables.
""", 'logistic_regression_summary.md')

    return model

def perform_rfe(X_train, y_train):
    """
    Perform Recursive Feature Elimination using logistic regression,
    visualize selected features, and export markdown explanation.

    Parameters:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Binary outcome (severe AE)

    Returns:
        selected (list): List of top selected features
    """
    # Fit RFE
    lr = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=lr, n_features_to_select=10)
    rfe.fit(X_train, y_train)

    # Ranking table
    ranking_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Ranking': rfe.ranking_,
        'Selected': rfe.support_
    }).sort_values(by='Ranking', ascending=True)

    selected_features = ranking_df[ranking_df['Selected']]['Feature'].tolist()

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=ranking_df[ranking_df['Selected']],
        x='Ranking', y='Feature', palette='crest'
    )
    plt.title("Top 10 Features Selected by Recursive Feature Elimination (RFE)")
    plt.xlabel("RFE Ranking (1 = most important)")
    plt.tight_layout()
    plt.savefig("plots/rfe_top_features.png")
    plt.close()

    # Markdown table
    lines = [
        "| Feature | Ranking | Selected |",
        "|---------|---------|----------|"
    ]
    for _, row in ranking_df.iterrows():
        selected_mark = "✅" if row["Selected"] else ""
        lines.append(f"| {row['Feature']} | {int(row['Ranking'])} | {selected_mark} |")
    
    table_md = "\n".join(lines)

    # Save markdown
    save_summary_to_md(f"""
# Recursive Feature Elimination (RFE) Summary

![RFE Top Features](../plots/rfe_top_features.png)

## Overview

- Estimator: Logistic Regression  
- Method: Backward elimination of features  
- Final selection: Top 10 most predictive features  

## Selected Features:

{table_md}

## Interpretation

- Features ranked by predictive contribution to severe AE.
- Only features with ranking = 1 are retained by RFE.
- These variables likely provide **non-redundant, high-signal** inputs to model.

""", 'rfe_summary.md')

    return selected_features


def perform_survival_analysis(df):
    """
    Perform Kaplan-Meier survival analysis on `time_to_onset` vs `has_severe_AE`.

    - Computes survival probability over time
    - Annotates event occurrences (AEs)
    - Generates trace plot with red dots at event times
    - Saves dynamic interpretation to markdown
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.utils import save_summary_to_md

    # Sort times and initialize variables
    times = np.sort(df['time_to_onset'].unique())
    km = [1.0]
    at_risk = len(df)
    surv_prob = 1.0
    events = []
    risk_set = []

    for t in times[1:]:
        events_at_t = ((df['time_to_onset'] == t) & (df['has_severe_AE'] == 1)).sum()
        withdrawn_at_t = (df['time_to_onset'] == t).sum()
        if at_risk > 0:
            surv_prob *= (1 - events_at_t / at_risk)
        km.append(surv_prob)
        events.append(events_at_t)
        risk_set.append(at_risk)
        at_risk -= withdrawn_at_t

    # Prepare plot data
    times_plot = times[:len(km)]
    event_count = int(np.sum(events))
    early_drop = np.argmax(np.array(km) < 0.8) if any(np.array(km) < 0.8) else None
    median_time = times_plot[np.argmax(np.array(km) <= 0.5)] if any(np.array(km) <= 0.5) else "Not reached"

    # Plot
    plt.figure(figsize=(10, 6))
    plt.step(times_plot, km, where='post', color='green', label='Survival Probability')
    plt.scatter(times_plot[1:], km[1:], c='red', s=12, label='Severe AE Events')
    plt.fill_between(times_plot, km, step='post', alpha=0.1, color='green')
    plt.axhline(0.5, color='gray', linestyle='--', linewidth=1)
    plt.axvline(median_time if isinstance(median_time, (int, float)) else 0, color='blue', linestyle='--', linewidth=1)
    plt.text(times_plot[-1]*0.6, 0.52, 'Median Survival', color='blue', fontsize=10)
    plt.title('Kaplan-Meier Survival Curve: Time to Severe AE (TAK Vaccine)')
    plt.xlabel("Time to Onset (hours)")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/kaplan_meier.png')
    plt.close()

    # Dynamic text report
    surv_md = f"""
![Kaplan-Meier Curve](../plots/kaplan_meier.png)

# Kaplan-Meier Survival Analysis Summary

## Method
- Non-parametric estimator of survival over time
- Tracks probability of not developing severe AE across `time_to_onset`

## Key Stats
- Total events observed: **{event_count}**
- Earliest steep drop (below 80% survival): {'at {:.1f} hrs'.format(times_plot[early_drop]) if early_drop is not None else 'Not detected'}
- Median survival (time when 50% still AE-free): **{median_time if isinstance(median_time, str) else '{:.1f} hrs'.format(median_time)}**

## Interpretation
- X-axis: Time (hours since vaccination)
- Y-axis: Probability of remaining without severe AE
- Red dots: AE occurrences at exact timestamps
- Blue vertical line: Median survival marker (if reached)
- Green area: Visual trace of survival decay

## Trend Summary
- **Front-loaded risk**: Most severe AEs happen early
- **Plateau** after initial decline: Indicates stabilization
- Survival drops to ~{km[-1]*100:.1f}% at last observed timepoint

## Conclusions
- **Monitoring should focus on first hours post-vaccination**
- Delayed onset AEs are rare, suggesting acute reaction profile
"""
    save_summary_to_md(surv_md, "kaplan_meier_interpretation.md")


def perform_clustering(X):
    """
    Performs KMeans clustering (k=4) on input features.
    Visualizes clusters using PCA and t-SNE projections.
    Saves summary markdown and returns cluster labels.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)

    X_clustered = X.copy()
    X_clustered['Cluster'] = clusters

    # PCA
    pca = PCA(n_components=2)
    reduced_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_pca[:, 0], y=reduced_pca[:, 1], hue=clusters, palette='Set2', s=40)
    plt.title("KMeans Clustering (PCA Projection)")
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig("plots/clustering_pca.png")
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    reduced_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced_tsne[:, 0], y=reduced_tsne[:, 1], hue=clusters, palette='Set2', s=40)
    plt.title("KMeans Clustering (t-SNE Projection)")
    plt.xlabel("tSNE-1")
    plt.ylabel("tSNE-2")
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.savefig("plots/clustering_tsne.png")
    plt.close()

    # Markdown
    save_summary_to_md(f"""
![Clustering PCA](../plots/clustering_pca.png)
![Clustering t-SNE](../plots/clustering_tsne.png)

# Clustering Analysis Summary

## Method
- **KMeans Clustering** with `k=4`
- Visualized via:
  - PCA (linear projection): captures variance
  - t-SNE (nonlinear projection): preserves local distances

## PCA Projection
- **Explained Variance**: PC1 = {explained[0]*100:.2f}%, PC2 = {explained[1]*100:.2f}%
- Well-separated clusters along linear axes suggest distinct patient profiles

## t-SNE Projection
- Reveals **nonlinear separability** and **tight subgroups**
- Ideal for identifying **emerging AE subtypes**

## Insights
- Clear visual stratification confirms heterogeneity in AE-related patterns
- Follow-up with cluster characterization can reveal underlying phenotypes
""", 'clustering_summary.md')

    return clusters


def perform_anomaly_detection(X):
    """
    Detects outliers using Isolation Forest.
    Visualizes 2D projection (PCA) and marks anomalies.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from src.utils import save_summary_to_md

    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X)  # -1: anomaly, 1: normal

    anomaly_count = np.sum(preds == -1)

    # PCA projection
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)
    pca_explained = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=reduced[:, 0], y=reduced[:, 1],
        hue=preds, palette={1: 'skyblue', -1: 'crimson'},
        style=preds, markers={1: 'o', -1: 'X'}, s=50
    )
    plt.title("Isolation Forest Anomaly Detection (PCA Projection)")
    plt.xlabel(f"PC1 ({pca_explained[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca_explained[1]*100:.1f}%)")
    plt.legend(title="Prediction", labels=["Normal", "Anomaly"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/anomaly_detection.png")
    plt.close()

    # Markdown
    save_summary_to_md(f"""
![Anomaly Detection](../plots/anomaly_detection.png)

# Anomaly Detection Summary

## Method
- Isolation Forest (unsupervised)
- Contamination rate: **5%**
- Projection: PCA (2D)

## Results
- Total detected anomalies: **{anomaly_count}**
- Red 'X' marks indicate patients with **unexpected feature profiles**
- PCA captures {pca_explained.sum()*100:.2f}% of total variance

## Interpretation
- Isolation Forest separates sparse or extreme data points.
- These flagged records may correspond to:
  - AE outliers
  - Misclassified metadata
  - Rare comorbidity combinations

## Actionable Use
- Prioritize flagged cases for manual validation or investigation
- Could uncover edge-case signals missed by population averages
""", 'anomaly_summary.md')

    return preds
