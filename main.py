import logging
from src.data_loader import load_data
from src.preprocessing import clean_data
from src.analysis import (
    analyze_lifestyle,
    analyze_physical_activity,
    analyze_education,
    analyze_sleep,
    analyze_sleep_100,
    analyze_bmi_vs_diagnosis_rate,
    analyze_sample_size
)

# Configure global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Alzheimer's Analysis Pipeline")
    
    # Execution steps
    data_path = "data/alzheimers_disease_data.csv"
    raw_df = load_data(data_path)
    clean_df = clean_data(raw_df)
    
    # Running Hypothesis tests
    analyze_sample_size(clean_df)
    analyze_lifestyle(clean_df)
    analyze_physical_activity(clean_df)
    analyze_education(clean_df)
    analyze_sleep(clean_df)
    analyze_sleep_100(clean_df)
    analyze_bmi_vs_diagnosis_rate(clean_df)

    logger.info("Analysis Pipeline finished successfully.")

if __name__ == "__main__":
    main()