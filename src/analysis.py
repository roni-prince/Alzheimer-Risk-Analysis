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
    Hypothesis: Investigating the link between Physical Activity and Alzheimer's Diagnosis.
    Uses Independent T-test for continuous data comparison.
    Note: Activity levels were recorded post-diagnosis.
    """
    
    # --- Data Distribution Analysis (Ensuring all 2,149 patients are included) ---
    # Using continuous ranges to account for decimal values
    bin1 = (df['PhysicalActivity'] < 4).sum()
    bin2 = ((df['PhysicalActivity'] >= 4) & (df['PhysicalActivity'] < 7)).sum()
    bin3 = (df['PhysicalActivity'] >= 7).sum()
    
    logger.info(f"Total patients in analysis: {len(df)}")
    logger.info(f"Group 1 (0 to <4 hours): {bin1} patients")
    logger.info(f"Group 2 (4 to <7 hours): {bin2} patients")
    logger.info(f"Group 3 (7 to 10 hours): {bin3} patients")

    # Statistical Calculation
    group_healthy = df[df['Diagnosis_bin'] == 0]['PhysicalActivity']
    group_alzheimer = df[df['Diagnosis_bin'] == 1]['PhysicalActivity']
    
    # nan_policy: handles how to treat missing (NaN) values in the data.
    _, p_val = stats.ttest_ind(group_healthy, group_alzheimer, nan_policy='omit')
    
    # Visualization
    plt.figure(figsize=(8, 5))
    
    sns.pointplot(
        data=df, 
        x='Diagnosis_bin', 
        y='PhysicalActivity', 
        hue='Diagnosis_bin', # Adds legend
        palette=[PALETTE[2], PALETTE[0]], # Blue for Healthy (2), Red for Alzheimer (0)
        join=False, # Don't connect categories to allow for independent comparison
        capsize=.1,
        scale=1.2 # Enlarges the dots size
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
    Hypothesis: Link between Education Level and MMSE scores.
    Uses ANOVA to compare means across multiple education groups.
    """
    # Group data by Education level for ANOVA
    edu_groups = [df[df['EducationLevel'] == i]['MMSE'] for i in sorted(df['EducationLevel'].unique())]
    _, p_val = stats.f_oneway(*edu_groups)
    #unique- includes all the indevidual MMSE scores be considered in the ANOVA.
    #sorted- for the data to be in order.
    
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
    
    # Changed to histplot with stat='percent' to show percentages
    # PALETTE[1] is Pink (for 0), PALETTE[2] is Blue (for 1)
    sns.histplot(
        data=df, 
        x='SleepQuality', 
        hue='MemoryComplaints_bin', 
        stat='percent', 
        common_norm=False, 
        kde=True,
        palette=[PALETTE[1], PALETTE[2]]
    )
    
    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"
    plt.title(f"Sleep Quality vs Memory Complaints\nT-test Result: {status} (p = {p_val:.4f})", fontweight='bold')
    
    # Ensure X-axis shows specifically 4 to 10
    plt.xlim(4, 10)
    plt.xticks([4, 5, 6, 7, 8, 9, 10])
    
    # Update axis labels to be clear
    plt.xlabel("Sleep Quality (4 to 10)")
    plt.ylabel("Percentage (%)")
    
    plt.show()
    
    logger.info(f"Sleep analysis complete. P-value: {p_val:.4f}")


def analyze_sleep_100(df: pd.DataFrame):
    """
    Hypothesis 3: Impact of Sleep Quality on Memory Complaints.
    Uses Independent T-test to compare two groups.
    """
    group_no = df[df['MemoryComplaints_bin'] == 0]['SleepQuality']
    group_yes = df[df['MemoryComplaints_bin'] == 1]['SleepQuality']
    _, p_val = stats.ttest_ind(group_no, group_yes)
    
    plt.figure(figsize=(8, 5))
    
    # Histplot with stat='percent'
    sns.histplot(
        data=df, 
        x='SleepQuality', 
        hue='MemoryComplaints_bin', 
        stat='percent', 
        common_norm=False, 
        kde=False, # <-- The curve is now removed
        palette=[PALETTE[1], PALETTE[2]] # Pink for No, Blue for Yes
    )
    
    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"
    plt.title(f"Sleep Quality vs Memory Complaints\nT-test Result: {status} (p = {p_val:.4f})", fontweight='bold')
    
    # Setting Y-axis from 0 to 100%
    plt.ylim(0, 100)
    
    # Ensure X-axis shows specifically 4 to 10
    plt.xlim(4, 10)
    plt.xticks([4, 5, 6, 7, 8, 9, 10])
    
    # Update axis labels
    plt.xlabel("Sleep Quality Score (4 to 10)")
    plt.ylabel("Percentage of Group (%)")
    
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Sleep analysis complete. P-value: {p_val:.4f}")


def analyze_bmi_vs_diagnosis_rate(df: pd.DataFrame):
    """
    Understanding the conenction between BMI and Alzheimer's diagnosis rate.
    Creates BMI groups, plots the percentage of diagnosed participants in each group,
    and calculates a chi-square p-value.
    """

    # Keep rows with valid BMI and diagnosis
    #This is a copy and we are not altering original dataset.
    #There are 4 bin categories: underweight, normal,over-weight and obese.
    bmi_df = df[['BMI', 'Diagnosis_bin']].dropna().copy()
    bmi_df['Diagnosis_bin'] = bmi_df['Diagnosis_bin'].astype(int)

    # Create BMI categories
    bmi_df['BMI_group'] = pd.cut(
        bmi_df['BMI'],
        bins=[0, 18.5, 25, 30, 100],
        labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
        right=False
    )

    # Summary for graph
    summary = bmi_df.groupby('BMI_group', observed=False)['Diagnosis_bin'].agg(['mean', 'count']).reset_index()
    #groups bmi_df by the created categories. Then it aggregates the diagnosis_bin, and provides the mean and the count of each category.
    #We have two goals here, we want to make sure that all of the rows were included in the calculation, and we want the average of each group.
    summary['Diagnosis_percent'] = summary['mean'] * 100
    #creates a new column: diagnosis_precent, we multiplied in order to be in the 0-100 scale.

    print("\nBMI and Alzheimer's Diagnosis Rate:")
    print(summary[['BMI_group', 'count', 'Diagnosis_percent']])

    # Chi-square test
    contingency_table = pd.crosstab(bmi_df['BMI_group'], bmi_df['Diagnosis_bin'])
    _, p_val, _, _ = stats.chi2_contingency(contingency_table)
    #we don't need all of the values, only the p_val.

    status = "Significant" if p_val < SIGNIFICANCE_LEVEL else "Not Significant"

    # Visualization
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary,
        x='BMI_group',
        y='Diagnosis_percent',
        hue='BMI_group',
        palette=PALETTE,
        legend=False
    )

    plt.title(
        f"Alzheimer's Diagnosis Rate by BMI Group\n"
        f"Result: {status} (p = {p_val:.4f})",
        fontweight='bold'
    )
    plt.xlabel("BMI Group")
    plt.ylabel("Alzheimer's Diagnosis Rate (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    plt.show()

    logger.info(
        f"BMI vs diagnosis rate analysis completed successfully. "
        f"Chi-square p-value: {p_val:.4f}"
    )


def analyze_sample_size(df: pd.DataFrame):
    #We decided to add a function that counts all the participants- meaning all the legitimate rows that were not empty.
    #this is after the preprocessing. The physical activity is not here because it does not go throught the preprocessing.

    print("\n" + "=" * 50)
    print("PARTICIPANT COUNT SUMMARY")
    print("=" * 50)

    total_n = len(df)
    print(f"Total number of participants: {total_n}")

    if 'Diagnosis_bin' in df.columns:
        diagnosis_counts = df['Diagnosis_bin'].value_counts(dropna=False).sort_index()
        healthy_n = diagnosis_counts.get(0, 0)
        alzheimer_n = diagnosis_counts.get(1, 0)

        print("\nDiagnosis distribution:")
        print(f"Healthy (0): {healthy_n} ({healthy_n / total_n:.1%})")
        print(f"Alzheimer's (1): {alzheimer_n} ({alzheimer_n / total_n:.1%})")
        #prints the precentage, with one number after the dot.

    if 'Smoking_bin' in df.columns:
        smoking_counts = df['Smoking_bin'].value_counts(dropna=False).sort_index()
        non_smokers = smoking_counts.get(0, 0)
        smokers = smoking_counts.get(1, 0)
        print("\nSmoking distribution:")
        print(f"Non-smokers (0): {non_smokers} ({non_smokers / total_n:.1%})")
        print(f"Smokers (1): {smokers} ({smokers / total_n:.1%})")

    if 'MemoryComplaints_bin' in df.columns:
        memory_counts = df['MemoryComplaints_bin'].value_counts(dropna=False).sort_index()
        no_memory = memory_counts.get(0, 0)
        yes_memory = memory_counts.get(1, 0)
        print("\nMemory complaints distribution:")
        print(f"No memory complaints (0): {no_memory} ({no_memory / total_n:.1%})")
        print(f"Memory complaints (1): {yes_memory} ({yes_memory / total_n:.1%})")

    if 'BMI' in df.columns and 'Diagnosis_bin' in df.columns:
        bmi_df = df[['BMI', 'Diagnosis_bin']].dropna().copy()

        bmi_df['BMI_group'] = pd.cut(
            bmi_df['BMI'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese'],
            right=False
        )
        #Here we want to see the number of rows- participants, in each BMI category.

        bmi_summary = bmi_df.groupby('BMI_group', observed=False)['Diagnosis_bin'].agg(['count', 'sum', 'mean'])

        print("\nBMI group distribution and diagnosis rate:")
        for group, row in bmi_summary.iterrows():
            group_n = int(row['count'])
            sick_n = int(row['sum'])
            sick_pct = row['mean'] * 100
            print(f"{group}: total={group_n}, Alzheimer's={sick_n} ({sick_pct:.1f}%)")

    print("=" * 50 + "\n")

    logger.info("Participant count summary completed successfully.")