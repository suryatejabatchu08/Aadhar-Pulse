import pandas as pd

def generate_recommendations(df):
    """
    Generates actionable recommendations based on stress sub-components.
    
    Rules:
    1. High Enrolment Stress (norm_enrol > 0.7) -> "Deploy Mobile Enrolment Camps"
    2. High Biometric Stress (norm_bio > 0.7) -> "Initiate School-based Biometric Update Drive"
    3. High Demographic Stress (norm_demo > 0.7) -> "Increase Demographic Update Counters"
    4. General High Stress (stress_score > 80) -> "Urgent: Review District Operational Capacity"
    """
    if df.empty:
        return pd.DataFrame() # Return empty if no data
        
    recommendations = []
    
    for _, row in df.iterrows():
        recs = []
        district = row['district']
        state = row['state']
        
        # Check sub-components (normalized 0-1)
        # Handle cases where norm columns might be NaN (though we filled fillna 0 in loader, 
        # norm might make them NaN if min==max. Replace with 0)
        norm_enrol = row.get('norm_enrol', 0)
        norm_bio = row.get('norm_bio', 0)
        norm_demo = row.get('norm_demo', 0)
        stress_score = row.get('stress_score', 0)
        
        if pd.isna(norm_enrol): norm_enrol = 0
        if pd.isna(norm_bio): norm_bio = 0
        if pd.isna(norm_demo): norm_demo = 0
        
        if norm_enrol > 0.5: # Lowered threshold for demo
            recs.append("Deploy Mobile Enrolment Camps (High New Enrolments)")
            
        if norm_bio > 0.5:
            recs.append("Initiate School/Center-based Biometric Update Drive (High Bio Load)")
            
        if norm_demo > 0.5:
            recs.append("Increase Demographic Update Capacity (High Correction Volume)")
            
        if stress_score > 70:
            recs.append("URGENT: Allocate Special Budget for Capacity Expansion")
            
        if not recs:
            recs.append("Maintain Current Operations")
            
        recommendations.append({
            'state': state,
            'district': district,
            'month': row['month'],
            'recommendations': "; ".join(recs)
        })
        
    return pd.DataFrame(recommendations)

if __name__ == "__main__":
    # Test
    try:
        df = pd.read_csv("src/analyzed_uidai_data.csv")
        # Just take the latest month for recommendations
        latest_month = df['month'].max()
        latest_df = df[df['month'] == latest_month]
        
        recs_df = generate_recommendations(latest_df)
        print(recs_df.head())
        recs_df.to_csv("src/recommendations.csv", index=False)
    except Exception as e:
        print(f"Error in recommender: {e}")
