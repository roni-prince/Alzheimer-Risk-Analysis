import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import logging

logger = logging.getLogger(__name__)
#allows log massages can be traced to their source.(Because there are a few files)

# Visual Settings
PALETTE = ["#b30047", "#ff4dc4", "#0000e6", "#9999ff"]
SIGNIFICANCE_LEVEL = 0.05

def analyze_lifestyle(df: pd.DataFrame):
    """
    Hypothesis 1: Investigating the link between Smoking and Alzheimer's Diagnosis.
    Uses Chi-Square test for categorical data.
    """
    # Statistical Calculation
    contingency = pd.crosstab(df['Smoking_bin'], df['Diagnosis_bin'])
    _, p_val, _, _ = stats.chi2_contingency(contingency)
    #diagnosis_bin- after the cleaning, we are left with a binary bin,
    #allows easy work with the data.
    #'_'- means that this value is there and exists, yet it is not in use by choice.
    
    # Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x='Smoking_bin', y='Diagnosis_bin', hue='Smoking_bin', palette=PALETTE[:2], legend=False)
    
    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"
    plt.title(f"Smoking vs Diagnosis\nResult: {status} (p = {p_val:.4f})", fontweight='bold')
    plt.xlabel("Smoking Status (0=No, 1=Yes)")
    plt.ylabel("AD Diagnosis Rate")
    plt.show()
    
    logger.info(f"Lifestyle analysis complete. P-value: {p_val:.4f}")
    #f- float, .4- four numbers after the dot.

def analyze_physical_activity(df: pd.DataFrame):
    """
    Hypothesis 4: Investigating the link between Physical Activity and Alzheimer's Diagnosis.
    Uses Independent T-test for continuous data comparison.
    Note: Activity levels were recorded post-diagnosis.
    """

    # Statistical Calculation
    group_healthy = df[df['Diagnosis_bin'] == 0]['PhysicalActivity']
    group_alzheimer = df[df['Diagnosis_bin'] == 1]['PhysicalActivity']
    
    _, p_val = stats.ttest_ind(group_healthy, group_alzheimer, nan_policy='omit')
    #nan_policy= what to do when there are missing (NaN) values in the data.
    
    # Visualization
    plt.figure(figsize=(8, 5))
    sns.pointplot(
        data=df, 
        x='Diagnosis_bin', 
        y='PhysicalActivity', 
        hue='Diagnosis_bin',#addes legend (מקרא)
        palette=[PALETTE[2], PALETTE[0]], # Blue for Healthy (2), Red for Alzheimer (0)
        join=False,#don't connect the two catagories, in order to be able to compare them.
        capsize=.1,
        scale=1.2#enlarges the dots size.
    )
    
    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"
    plt.title(f"Physical Activity by Diagnosis\nResult: {status} (p = {p_val:.4f})", fontweight='bold')
    plt.xticks([0, 1], ['Healthy', 'Alzheimer'])
    plt.xlabel("Diagnosis Group")
    plt.ylabel("Average Physical Activity Score")
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Physical activity analysis complete. P-value: {p_val:.4f}")
    
def analyze_education(df: pd.DataFrame):
    """
    Hypothesis 2: Link between Education Level and MMSE scores.
    Uses ANOVA to compare means across multiple education groups.
    """
    # Group data by Education level for ANOVA
    edu_groups = [df[df['EducationLevel'] == i]['MMSE'] for i in sorted(df['EducationLevel'].unique())]
    _, p_val = stats.f_oneway(*edu_groups)
    
    # Visualization
    plt.figure(figsize=(8, 5))
    sns.pointplot(data=df, x='EducationLevel', y='MMSE', color=PALETTE[0], capsize=.1)
    
    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"
    plt.title(f"Average MMSE by Education Level\nANOVA Result: {status} (p = {p_val:.4f})", fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.show()
    
    logger.info(f"Education analysis complete. P-value: {p_val:.4f}")

def analyze_sleep(df: pd.DataFrame):
    """
    Hypothesis 3: Impact of Sleep Quality on Memory Complaints.
    Uses Independent T-test to compare two groups.
    """
    group_no = df[df['MemoryComplaints_bin'] == 0]['SleepQuality']
    group_yes = df[df['MemoryComplaints_bin'] == 1]['SleepQuality']
    _, p_val = stats.ttest_ind(group_no, group_yes)
    
    # Visualization
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df, x='SleepQuality', hue='MemoryComplaints_bin', fill=True, palette=PALETTE[2:])
    
    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"
    plt.title(f"Sleep Quality vs Memory Complaints\nT-test Result: {status} (p = {p_val:.4f})", fontweight='bold')
    plt.show()
    
    logger.info(f"Sleep analysis complete. P-value: {p_val:.4f}")