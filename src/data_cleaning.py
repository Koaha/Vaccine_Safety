import pandas as pd
import numpy as np
from datetime import datetime
from scipy.spatial import KDTree
from src.utils import logger
from sklearn.ensemble import IsolationForest

def load_and_filter_data(file_path):
    """Load CSV data and filter to the most common vaccine, renaming to 'TAK'."""
    df = pd.read_csv(file_path)
    most_common_vaccine = df['vaccine_1_name'].mode()[0]
    df = df[df['vaccine_1_name'] == most_common_vaccine].copy().reset_index(drop=True)
    df['vaccine_1_name'] = 'TAK'
    logger.info(f"Filtered to vaccine 'TAK' (originally {most_common_vaccine}). Shape: {df.shape}")
    return df

def clean_data(df, categorical_cols, numerical_cols, date_cols, text_cols):
    """Convert types, handle missing values, check consistency, logic, and outliers."""
    # Convert types
    df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric, errors='coerce')
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        logger.debug(f"Converted {col} to datetime.")

    # Handle missing values
    for col in categorical_cols:
        df[col] = df[col].fillna('Unknown').astype(str)
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median(skipna=True))
        logger.debug(f"Filled missing in {col} with median: {df[col].median():.2f}")
    for col in text_cols:
        df[col] = df[col].fillna('No description')
    for col in date_cols:
        df[col] = df[col].fillna(method='ffill')
    all_dates_missing = df[date_cols].isnull().all(axis=1)
    logger.info(f"Dropping {all_dates_missing.sum()} rows with all dates missing.")
    df = df[~all_dates_missing].reset_index(drop=True)

    # Check consistency (boundaries)
    df['age'] = df['age'].clip(0, 120)
    df['onset_hour'] = df['onset_hour'].clip(0, None)
    logger.info("Applied boundary clipping to age and onset_hour.")

    # Check logic and conflicts
    invalid_gender = ((df['female'] == '1') & (df['male'] == '1')) | ((df['female'] == '0') & (df['male'] == '0'))
    logger.warning(f"Found {invalid_gender.sum()} invalid gender rows. Setting to unknown ('0','0').")
    df.loc[invalid_gender, ['female', 'male']] = ['0', '0']

    df.loc[df['so_mui_vaccine'] < 2, ['vaccine_2_name', 'vaccine_2_dose_number', 'vaccine_2_hour', 'vaccine_2_date', 'vaccine_2_lot_number']] = np.nan

    df['valid_timing'] = df['onset_date'] > df['vaccine_1_date']
    invalid_timing = ~df['valid_timing']
    logger.warning(f"Found {invalid_timing.sum()} invalid timing rows. Dropping them.")
    df = df[~invalid_timing].reset_index(drop=True)

    if df.index.duplicated().any():
        logger.warning("Found duplicate indices after timing filter, resetting...")
        df = df.reset_index(drop=True)

    # Outliers handling
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
        logger.debug(f"Outliers in {col}: {outliers.sum()}")
        df.loc[outliers, col] = df[col].median()

    # Isolation Forest for anomalies
    iso = IsolationForest(contamination=0.05, random_state=42)
    df['outlier'] = iso.fit_predict(df[numerical_cols].fillna(0))
    logger.info(f"Detected {(df['outlier'] == -1).sum()} outliers.")

    return df