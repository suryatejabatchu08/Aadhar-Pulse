import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def calculate_stress_index(master_df):
    """
    Calculates the 'Aadhaar Stress Index' for each district-month.
    
    Formula Factors:
    1. Biometric Stress: High volume of bio updates, esp for age 5-17 (mandatory).
    2. Demographic Stress: High volume of corrections.
    3. Enrolment Growth: High new enrolments (indicates population pressure).
    
    Returns df with 'stress_score', 'risk_category'.
    """
    if master_df.empty:
        return master_df
    
    df = master_df.copy()
    
    # 1. Feature Engineering
    # Biometric Pressure: Focus on age 5-17 (Mandatory updates) and general load
    # If columns exist like 'bio_bio_age_5_17', use them. Otherwise sum all bio columns.
    bio_cols = [c for c in df.columns if 'bio_' in c and c not in ['bio_state', 'bio_district']]
    demo_cols = [c for c in df.columns if 'demo_' in c and c not in ['demo_state', 'demo_district']]
    enrol_cols = [c for c in df.columns if 'enrol_' in c and c not in ['enrol_state', 'enrol_district']]
    
    df['total_bio_updates'] = df[bio_cols].sum(axis=1)
    df['total_demo_updates'] = df[demo_cols].sum(axis=1)
    df['total_enrolments'] = df[enrol_cols].sum(axis=1)
    
    # Identify specific mandatory update columns (heuristic matching)
    # The loader prefixes them with 'bio_'. The csv cols were 'age_5_17' etc.
    # So look for 'bio_age_5_17' or similar.
    # We will use total_bio for robustness if specific cols missing.
    
    # 2. Normalization (0-1 Scale) relative to the entire dataset (or per state?)
    # Let's normalize across the whole dataset to find "National Hotspots".
    scaler = MinMaxScaler()
    
    cols_to_norm = ['total_bio_updates', 'total_demo_updates', 'total_enrolments']
    norm_cols = ['norm_bio', 'norm_demo', 'norm_enrol']
    
    df[norm_cols] = scaler.fit_transform(df[cols_to_norm])
    
    # 3. Weighted Score
    # Weights: Biometric (40% - operationally heavy), Demo (30%), Enrolment (30%)
    w_bio = 0.4
    w_demo = 0.3
    w_enrol = 0.3
    
    df['stress_score_raw'] = (
        w_bio * df['norm_bio'] + 
        w_demo * df['norm_demo'] + 
        w_enrol * df['norm_enrol']
    )
    
    # Scale to 0-100
    df['stress_score'] = df['stress_score_raw'] * 100
    
    # 4. Risk Categorization
    def categorize_risk(score):
        if score > 80: return 'Red'
        if score > 50: return 'Amber'
        return 'Green'
    
    df['risk_category'] = df['stress_score'].apply(categorize_risk)
    
    return df

if __name__ == "__main__":
    # Test
    try:
        df = pd.read_csv("src/processed_uidai_data_sample.csv")
        scored_df = calculate_stress_index(df)
        print(scored_df[['state', 'district', 'stress_score', 'risk_category']].head())
        scored_df.to_csv("src/scored_uidai_data.csv", index=False)
    except Exception as e:
        print(f"Error checking stress index: {e}")
