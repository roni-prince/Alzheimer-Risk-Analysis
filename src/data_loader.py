import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_data(path: str) -> pd.DataFrame:
    #Loads the dataset from a CSV file path.
    try:
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded successfully with {df.shape[0]} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise