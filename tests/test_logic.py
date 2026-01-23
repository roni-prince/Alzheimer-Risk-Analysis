import pytest
import pandas as pd
import numpy as np
from src.preprocessing import clean_data

def test_cleaning_logic():
    """
    Test if the cleaning function correctly encodes 
    'Yes' and 'No' strings into 1 and 0.
    """
    # Create a small "fake" dataset for testing
    mock_data = {
        'Smoking': ['Yes', 'No', 'yes', 'no'],
        'Diagnosis': [1, 0, 1, 0],
        'MemoryComplaints': ['Yes', 'No', 'No', 'Yes']
    }
    df = pd.DataFrame(mock_data)
    
    # Run the cleaning function
    cleaned_df = clean_data(df)
    
    # Check if the mapping worked correctly
    assert cleaned_df['Smoking_bin'].iloc[0] == 1
    assert cleaned_df['Smoking_bin'].iloc[1] == 0
    assert 'Diagnosis_bin' in cleaned_df.columns
    
def test_missing_values_handling():
    """
    Test if the function correctly handles or removes 
    rows with missing Diagnosis values.
    """
    mock_data = {
        'Smoking': ['Yes', 'No'],
        'Diagnosis': [1, np.nan] # One valid, one missing
    }
    df = pd.DataFrame(mock_data)
    
    cleaned_df = clean_data(df)
    
    # After cleaning, the row with NaN Diagnosis should be gone
    assert len(cleaned_df) == 1