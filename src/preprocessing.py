import pandas as pd
import logging
#logging is used for writing messages, like errors, warnings, info.

logger = logging.getLogger(__name__)
#The logger is the object used for writing the info messages without print().
#The reason behind this is that it is more controllable, this way we can send the logs it made to an external file.
#Here we are giving the object it's name- logger.

# Standard mapping for text-based categories
#allows us to handle different variation which we may encounter in the data set.
BINARY_MAP = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0}

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #Standardizes columns and encodes variables for analysis.
    #The copy is of the dataframe, we do not want to change the original data, but we do want a new copy to ba able to modify it fo our liking.
    #This way, we will be able to prevent mistakes  and misunderstandings
    df = df.copy()
    
    #1.Target Variable
    #convert to numeric- this way we will be able to calculate.
    #errors='coerce' is important when the column contains strings like "?" or "unknown".
    if 'Diagnosis' in df.columns:
        df['Diagnosis_bin'] = pd.to_numeric(df['Diagnosis'], errors='coerce')
    
    #2.Lifestyle & Symptoms (Smoking, MemoryComplaints)
    #Switch them to binary format.
    #Also, if th ecolumn is already numeric and there are NaNs, avoid turning them into strings.
    for col in ['Smoking', 'MemoryComplaints']:
        if col in df.columns:
            # If the column is already numeric (0/1), just copy it
            if pd.api.types.is_numeric_dtype(df[col]):
                df[f"{col}_bin"] = df[col]
            else:
                #If it's text, convert to lowercase and map to 0/1
                #Normalizes the whitespaces and the text. 
                df[f"{col}_bin"] = df[col].astype(str).str.lower().str.strip().map(BINARY_MAP)
            
    # Remove rows without a valid diagnosis
    df = df.dropna(subset=['Diagnosis_bin'])
    df['Diagnosis_bin'] = df['Diagnosis_bin'].astype(int)     # Ensure Diagnosis_bin is numeric to avoid plotting warnings
    
    logger.info("Data preprocessing completed successfully.")
    #logs a success message
    return df

