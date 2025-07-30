from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.utils import logger
import pandas as pd

def engineer_features(df, categorical_cols, numerical_cols):
    """Create new features, encode categoricals, process text, normalize."""
    # Temporal features
    df['time_to_onset'] = (df['onset_date'] - df['vaccine_1_date']).dt.days * 24 + df['onset_hour']
    df['ae_duration'] = df['ket_thuc_time']
    logger.debug("Created time_to_onset and ae_duration.")

    # Comorbidity index
    allergy_cols = ['di_ung_thuoc', 'di_ung_thuc_an', 'di_ung_vaccine', 'di_ung_khac']
    df['has_allergy'] = df[allergy_cols].apply(pd.to_numeric, errors='coerce').max(axis=1)
    logger.debug("Created has_allergy index.")

    # One-hot encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[['vaccine_1_name', 'vung_yk']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=df.index)
    df = pd.concat([df, encoded_df], axis=1)
    logger.debug("Applied one-hot encoding to vaccine_1_name and vung_yk.")

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
    tfidf = vectorizer.fit_transform(df['mo_ta_dien_bien'])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=[f'tfidf_{name}' for name in vectorizer.get_feature_names_out()], index=df.index)
    df = pd.concat([df, tfidf_df], axis=1)
    logger.debug("Applied TF-IDF to mo_ta_dien_bien.")

    # Sentence Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['mo_ta_dien_bien'].tolist(), show_progress_bar=True)
    embed_df = pd.DataFrame(embeddings, columns=[f'text_emb_{i}' for i in range(embeddings.shape[1])], index=df.index)
    df = pd.concat([df, embed_df], axis=1)

    # Normalization
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    logger.debug("Normalized numerical columns.")

    return df