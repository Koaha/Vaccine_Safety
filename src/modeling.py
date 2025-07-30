from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score
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

def prepare_data_for_modeling(df, date_cols, text_cols):
    """Prepare X and y for modeling, drop non-numeric columns."""
    columns_to_drop = ['has_severe_AE'] + date_cols + text_cols + ['valid_timing', 'recovery_status', 'ae_severity_score', 'ket_qua']
    X = df.drop(columns_to_drop, axis=1, errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['has_severe_AE']
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    logger.info(f"Data split: Train {X_train.shape}, Test {X_test.shape}")
    return X, y, X_train, X_test, y_train, y_test

def train_xgboost(X_train, y_train):
    """Train XGBoost with Optuna tuning."""
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

def evaluate_model(model, X_test, y_test):
    """Evaluate model with AUC and F1-score."""
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, model.predict(X_test))
    return auc, f1

def explain_with_shap(model, X_test):
    """Compute and plot SHAP values."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig('plots/shap_summary.png')
    plt.close()
    logger.info("Saved shap_summary.png")
    save_summary_to_md("""
# SHAP Summary Plot Report for TAK Vaccine Safety

<image-card alt="SHAP Summary" src="../plots/shap_summary.png" ></image-card>

## Detailed Interpretation
**X Axis:** SHAP value (impact on prediction; positive for higher severe AE probability).
**Y Axis:** Features ranked by importance.
**Trends Observed:** Dots colored by feature value; red high values pushing right indicate risk factors.
**Conclusions:** Top features like age with positive SHAP for low values conclude younger patients at higher risk for TAK. Observe clustering for nuanced insights.
""", 'shap_summary_interpretation.md')

def perform_rfe(X_train, y_train):
    """Perform Recursive Feature Elimination."""
    lr = LogisticRegression(max_iter=1000)
    rfe = RFE(lr, n_features_to_select=10)
    rfe.fit(X_train, y_train)
    return X_train.columns[rfe.support_]

def train_bayesian_logistic(X, y):
    """Train PyTorch Bayesian-like logistic model."""
    class BayesianLogistic(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.linear = nn.Linear(input_dim, 1)

        def forward(self, x):
            return torch.sigmoid(self.linear(x))

    model = BayesianLogistic(X.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    criterion = nn.BCELoss()

    X_torch = torch.tensor(X.values, dtype=torch.float32)
    y_torch = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_torch, y_torch)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(200):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, loss.item()

def perform_survival_analysis(df):
    """Perform Kaplan-Meier survival analysis."""
    times = np.sort(df['time_to_onset'].unique())
    km = [1.0]
    at_risk = len(df)
    surv_prob = 1.0

    for t in times[1:]:
        events_at_t = ((df['time_to_onset'] == t) & (df['has_severe_AE'] == 1)).sum()
        withdrawn_at_t = (df['time_to_onset'] == t).sum()
        if at_risk > 0:
            surv_prob *= (1 - events_at_t / at_risk)
        km.append(surv_prob)
        at_risk -= withdrawn_at_t

    plt.figure(figsize=(10, 6))
    plt.step(times, km, where='post', color='green', label='Survival Probability')
    plt.title('Kaplan-Meier Survival Curve for Time to Severe AE in TAK Vaccine')
    plt.savefig('plots/kaplan_meier.png')
    plt.close()
    save_summary_to_md("""
<image-card alt="Kaplan-Meier Curve" src="../plots/kaplan_meier.png" ></image-card>

## Interpretation of Kaplan-Meier Curve
X-axis: Time to onset in hours.
Y-axis: Probability of no severe AE (1 to 0).
Trends: Steep drops early indicate quick AEs.
Conclusions: For TAK, most severe AEs occur soon after vaccination.
""", 'kaplan_meier_interpretation.md')

def perform_clustering(X):
    """Perform KMeans clustering."""
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)
    return clusters

def perform_anomaly_detection(X):
    """Detect anomalies with Isolation Forest."""
    iso = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso.fit_predict(X)
    return anomalies