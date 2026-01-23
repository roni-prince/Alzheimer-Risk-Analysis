import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Constants for standardized mapping
BINARY_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns and encodes categorical variables."""
    df = df.copy()
    
    # Encode target variable
    if 'Diagnosis' in df.columns:
        df['Diagnosis_bin'] = pd.to_numeric(df['Diagnosis'], errors='coerce')
    
    # Encode lifestyle features
    for col in ['Smoking', 'MemoryComplaints']:
        if col in df.columns:
            df[f"{col}_bin"] = df[col].astype(str).str.lower().str.strip().map(BINARY_MAP)
            
    df = df.dropna(subset=['Diagnosis_bin'])
    logger.info("Data preprocessing completed successfully.")
    return df