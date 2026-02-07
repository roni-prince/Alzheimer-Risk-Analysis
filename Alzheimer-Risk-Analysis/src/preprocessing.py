import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Standard mapping for text-based categories
BINARY_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes columns and encodes variables for analysis."""
    df = df.copy()
    
    # 1. Target Variable
    if 'Diagnosis' in df.columns:
        df['Diagnosis_bin'] = pd.to_numeric(df['Diagnosis'], errors='coerce')
    
    # 2. Lifestyle & Symptoms (Smoking, MemoryComplaints)
    for col in ['Smoking', 'MemoryComplaints']:
        if col in df.columns:
            # If the column is already numeric (0/1), just copy it
            if pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_bin"] = df[col]
            else:
                # If it's text, convert to lowercase and map to 0/1
                df[f"{col}_bin"] = df[col].astype(str).str.lower().str.strip().map(BINARY_MAP)
            
    # Remove rows without a valid diagnosis
    df = df.dropna(subset=['Diagnosis_bin'])
    
    logger.info("Data preprocessing completed successfully.")
    return df