from scipy.spatial import KDTree
import statsmodels.api as sm
from econml.dml import CausalForestDML
from doubleml import DoubleMLData, DoubleMLPLR
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import logger, save_summary_to_md
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def perform_psm(df, confounders):
    """Perform Propensity Score Matching."""
    X_ps = df[confounders]
    ps_model = LogisticRegression(max_iter=1000)
    ps_model.fit(X_ps, df['has_allergy'])
    df['ps'] = ps_model.predict_proba(X_ps)[:, 1]

    treated = df[df['has_allergy'] == 1]
    control = df[df['has_allergy'] == 0]

    tree = KDTree(control[['ps']])
    dist, ind = tree.query(treated[['ps']], k=1)
    matched_control = control.iloc[ind.flatten()]
    matched = pd.concat([treated, matched_control])

    ate = matched[matched['has_allergy'] == 1]['has_severe_AE'].mean() - matched[matched['has_allergy'] == 0]['has_severe_AE'].mean()
    logger.info(f"ATE from PSM (allergy on severe AE): {ate:.4f}")

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(df['ps'], kde=True, color='blue', label='All')
    sns.histplot(matched['ps'], kde=True, color='green', label='Matched')
    plt.title('Propensity Score Distribution Before and After Matching')
    plt.savefig('plots/psm_distribution.png')
    plt.close()
    save_summary_to_md("""
# PSM Causal Inference Results

ATE: {ate:.4f}

<image-card alt="PSM Distribution" src="../plots/psm_distribution.png" ></image-card>
""", 'psm_results.md')

    return ate

def perform_tmle(X, y):
    """Perform Targeted Maximum Likelihood Estimation."""
    try:
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
        logger.info(f"Approximate TMLE ATE: {ate_tmle:.4f}")
        return ate_tmle
    except Exception as e:
        logger.warning(f"TMLE failed: {e}")
        return np.nan

def perform_causal_forest(confounders, treatment, outcome):
    """Perform Causal Forest."""
    est = CausalForestDML(model_y=RandomForestRegressor(), 
                        model_t=RandomForestRegressor(),
                        n_estimators=100, 
                        min_samples_leaf=10, 
                        random_state=42)
    est.fit(Y=outcome, T=treatment, X=confounders)
    ate_cf = est.const_marginal_ate(confounders.mean().values[None, :])[0][0]
    logger.info(f"Causal Forest ATE: {ate_cf:.4f}")
    cate_cf = est.effect(confounders)
    plt.hist(cate_cf, bins=30)
    plt.savefig('plots/causal_forest_te.png')
    plt.close()
    save_summary_to_md(f"ATE: {ate_cf:.4f}", 'causal_forest_results.md')
    return ate_cf

def perform_double_ml(confounders, treatment, outcome):
    """Perform DoubleML."""
    df_dml = pd.concat([confounders, treatment.to_frame(), outcome.to_frame()], axis=1).dropna()
    X_dml = df_dml[confounders.columns]
    y_dml = df_dml[outcome.name].values
    d_dml = df_dml[treatment.name].values

    dml_data = DoubleMLData.from_arrays(X_dml.values, y_dml, d_dml, x_cols=list(X_dml.columns))
    ml_g = RandomForestClassifier(n_estimators=100, random_state=42)
    ml_m = RandomForestClassifier(n_estimators=100, random_state=42)
    dml_model = DoubleMLPLR(dml_data, ml_g, ml_m, n_folds=5)
    dml_model.fit()
    ate_dml = dml_model.coef[0]
    ate_dml_ci = dml_model.confint(level=0.95)
    logger.info(f"DoubleML ATE: {ate_dml:.4f}")
    plt.hist(np.random.normal(ate_dml, (ate_dml_ci.iloc[0, 1] - ate_dml) / 1.96, 1000), bins=30)
    plt.savefig('plots/dml_ate_bootstrap.png')
    plt.close()
    save_summary_to_md(f"ATE: {ate_dml:.4f}", 'doubleml_results.md')
    return ate_dml

# Comment out DoWhy if too slow
# def perform_dowhy(df, confounders):
#     common_causes = [c for c in confounders.columns if c not in ['has_allergy', 'has_severe_AE']]
#     model = CausalModel(data=df, treatment='has_allergy', outcome='has_severe_AE', common_causes=common_causes)
#     identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
#     causal_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.propensity_score_matching")
#     ate_dowhy = causal_estimate.value
#     logger.info(f"DoWhy ATE: {ate_dowhy:.4f}")
#     return ate_dowhy