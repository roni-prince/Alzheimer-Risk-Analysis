# Alzheimer's Disease Analysis & Statistical System

## Research Paper
You can read the mini-paper [here on Google Drive](https://docs.google.com/document/d/1r2p0a9bw7OTt7JlBNB4j6_qb4v0_A1BGfarhBorS0LU/edit?usp=sharing).

## 1. Project Description & Objectives
**Description:** This project analyzes the impact of lifestyle habits, education, and clinical indicators on Alzheimer's Disease diagnosis. The system utilizes a **modular data pipeline** to process clinical records, perform statistical validation, and visualize key risk factors.

**Main Objectives:**
* Identify significant correlations between **lifestyle factors** (Smoking, Physical Activity) and Alzheimer's diagnosis.
* Investigate the **"protective effect"** of education on cognitive performance (MMSE scores).
* Analyze the relationship between **sleep quality** and subjective memory complaints.

**Hypotheses:**
* **Lifestyle:** We hypothesize that smoking and low physical activity are significantly correlated with a higher probability of diagnosis.
* **Education:** We expect a positive correlation where higher education levels align with higher MMSE scores.
* **Sleep:** We hypothesize that poor sleep quality is a primary driver for memory complaints in non-diagnosed individuals.

---

## 2. Project Structure
The project is organized into a modular structure to ensure readability and professional code management:

* **`main.py`**: The entry point that orchestrates the entire pipeline.
* **`src/`**: 
    * `data_loader.py`: Handles CSV ingestion and initial logging.
    * `preprocessing.py`: Cleaning, encoding, and integrity checks.
    * `analysis.py`: Executes statistical tests (Chi-Square, T-Test, ANOVA) and plotting functions for hypotheses.
* **`tests/`**: Contains automated unit tests (Pytest) to ensure data integrity.
* **`data/`**: Directory for the dataset file.
* **`requirements.txt`**: List of project dependencies.

---

## 3. Key Stages & Workflow
The project follows a structured data analysis pipeline:

1. **Data Import:** Loading the clinical dataset using `data_loader.py` with structured logging.
2. **Feature Engineering & Cleaning:**
    * Mapping categorical "Yes/No" strings to **Binary Encoding** (0/1).
    * Handling outliers in health metrics (BMI, Sleep Quality).
    * Ensuring data integrity by removing incomplete records.
3. **Statistical Analysis:**
    * **Chi-Square Tests:** For categorical relationships (Smoking vs. Diagnosis).
    * **Independent T-Tests:** For comparing Sleep Quality means.
    * **ANOVA:** For multi-group Education Level analysis.
4. **Visualization:** Generating KDE plots, Point plots, and Bar charts to interpret findings.

---

## 4. Dataset Description
* **Source:** Alzheimer's Disease Dataset (Kaggle).
* **Target Variable:** `Diagnosis` (0 = Healthy, 1 = Alzheimer's).
* **Key Features:** MMSE, EducationLevel, Smoking, SleepQuality, PhysicalActivity.

---

## 5. How to Run
To run this project on your local machine, follow these steps:

**Step 1: Install Dependencies**
```bash
pip install -r requirements.txt 
```
**Step 2: Run the Analysis**
```bash
python main.py
```
**Step 3: Run Automated Tests**
```bash
pytest tests/
```

---

## 6. References
* **Kaggle Dataset:** [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)