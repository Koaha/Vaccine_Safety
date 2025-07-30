import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_cleaning import load_and_filter_data, clean_data
from EDA import perform_eda
from feature_engineering import engineer_features
from modeling import prepare_data_for_modeling, train_xgboost, evaluate_model, explain_with_shap, perform_rfe, train_bayesian_logistic, perform_survival_analysis, perform_clustering, perform_anomaly_detection
from causal_inference import perform_psm, perform_tmle, perform_causal_forest, perform_double_ml
from utils import logger

# Define columns (copy from original code)
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

# Load and clean
df = load_and_filter_data('dataset/synthetic_vaccine_sae_data.csv')
df = clean_data(df, categorical_cols, numerical_cols, date_cols, text_cols)

# EDA
perform_eda(df, numerical_cols, categorical_cols)

# Feature Engineering
df = engineer_features(df, categorical_cols, numerical_cols)

# Modeling
X, y, X_train, X_test, y_train, y_test = prepare_data_for_modeling(df, date_cols, text_cols)
xgb_model, best_params = train_xgboost(X_train, y_train)
auc, f1 = evaluate_model(xgb_model, X_test, y_test)
explain_with_shap(xgb_model, X_test)
selected_features = perform_rfe(X_train, y_train)
bayesian_model, loss = train_bayesian_logistic(X, y)
perform_survival_analysis(df)
clusters = perform_clustering(X)
anomalies = perform_anomaly_detection(X)

# Causal Inference
perform_psm(df, ['age', 'so_mui_vaccine', 'time_to_onset'])
perform_tmle(X, y)
perform_causal_forest(confounders=X.copy(), 
                    treatment=df['has_allergy'], 
                    outcome=df['has_severe_AE'])
perform_double_ml(X, df['has_allergy'], df['has_severe_AE'])

logger.info("Pipeline complete.")