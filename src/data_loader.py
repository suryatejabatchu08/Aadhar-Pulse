import pandas as pd
import glob
import os

# Define base paths
BASE_PATH = os.path.join(os.path.dirname(__file__), "..", "Datasets")

ENROLMENT_PATH = os.path.join(BASE_PATH, "api_data_aadhar_enrolment")
DEMO_UPDATE_PATH = os.path.join(BASE_PATH, "api_data_aadhar_demographic")
BIO_UPDATE_PATH = os.path.join(BASE_PATH, "api_data_aadhar_biometric")

def load_and_merge_csvs(folder_path, dataset_type):
    """
    Loads all CSVs from a folder and merges them.
    Adds a 'type' column to distinguish datasets if needed, 
    but primarily ensures column consistency.
    """
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        print(f"Warning: No files found in {folder_path}")
        return pd.DataFrame()

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    
    # Standardize column names
    full_df.columns = [c.lower().strip() for c in full_df.columns]
    
    # Convert date to datetime
    if 'date' in full_df.columns:
        full_df['date'] = pd.to_datetime(full_df['date'], format='%d-%m-%Y', errors='coerce')
        
    # Standardize string casing for categorical columns
    if 'state' in full_df.columns:
        full_df = clean_state_names(full_df)
        
    if 'district' in full_df.columns:
        full_df['district'] = full_df['district'].astype(str).str.title().str.strip()
        # Remove rows where district is numeric
        full_df = full_df[~full_df['district'].str.isnumeric()]
        
    return full_df

def clean_state_names(df):
    """
    Standardizes state names and filters out garbage/cities.
    """
    df['state'] = df['state'].astype(str).str.title().str.strip()
    
    # 1. Remove numeric
    df = df[~df['state'].str.isnumeric()]
    
    # 2. Mapping Dictionary
    mapping = {
        'Andaman & Nicobar Islands': 'Andaman And Nicobar Islands',
        'Dadra & Nagar Haveli': 'Dadra And Nagar Haveli',
        'Daman & Diu': 'Daman And Diu',
        'Jammu & Kashmir': 'Jammu And Kashmir',
        'Orissa': 'Odisha',
        'Pondicherry': 'Puducherry',
        'West  Bengal': 'West Bengal',
        'West Bangal': 'West Bengal',
        'Westbengal': 'West Bengal',
        'West Bengli': 'West Bengal',
        'Chhatisgarh': 'Chhattisgarh',
        'Tamilnadu': 'Tamil Nadu',
        'Uttaranchal': 'Uttarakhand',
        'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra And Nagar Haveli And Daman And Diu',
    }
    
    df['state'] = df['state'].replace(mapping)
    
    # 3. Whitelist (Valid States & UTs) to remove cities/garbage
    # This list covers all standard Indian States/UTs + merged UTs
    valid_states = [
        'Andaman And Nicobar Islands', 'Andhra Pradesh', 'Arunachal Pradesh', 'Assam', 'Bihar', 
        'Chandigarh', 'Chhattisgarh', 'Dadra And Nagar Haveli', 'Daman And Diu', 
        'Dadra And Nagar Haveli And Daman And Diu', 'Delhi', 'Goa', 'Gujarat', 'Haryana', 
        'Himachal Pradesh', 'Jammu And Kashmir', 'Jharkhand', 'Karnataka', 'Kerala', 
        'Ladakh', 'Lakshadweep', 'Madhya Pradesh', 'Maharashtra', 'Manipur', 'Meghalaya', 
        'Mizoram', 'Nagaland', 'Odisha', 'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 
        'Tamil Nadu', 'Telangana', 'Tripura', 'Uttar Pradesh', 'Uttarakhand', 'West Bengal'
    ]
    
    # Filter to only valid states
    df = df[df['state'].isin(valid_states)]
    
    return df

def get_monthly_level_data():
    """
    Loads all three datasets, aggregates them to (Month, State, District) level,
    and returns a consolidated dataframe for analysis.
    """
    print("Loading Enrolment Data...")
    df_enrol = load_and_merge_csvs(ENROLMENT_PATH, "enrolment")
    
    print("Loading Demographic Update Data...")
    df_demo = load_and_merge_csvs(DEMO_UPDATE_PATH, "demographic")
    
    print("Loading Biometric Update Data...")
    df_bio = load_and_merge_csvs(BIO_UPDATE_PATH, "biometric")
    
    # Define aggregation columns
    group_cols = ['date', 'state', 'district']
    
    # Helper to aggregate
    def aggregate_dataset(df, prefix):
        if df.empty:
            return pd.DataFrame()
        
        # Ensure we don't lose data by grouping on Date (Day) -> convert to Month first?
        # Actually, let's keep daily first or convert to Month start
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        
        # Identify numeric columns for sum
        numeric_cols = [c for c in df.columns if c not in group_cols + ['month', 'pincode']]
        
        # Group by Month, State, District
        agg_df = df.groupby(['month', 'state', 'district'])[numeric_cols].sum().reset_index()
        
        # Rename columns with prefix to avoid collision
        agg_df.columns = ['month', 'state', 'district'] + [f"{prefix}_{c}" for c in numeric_cols]
        return agg_df

    # Aggregate each
    agg_enrol = aggregate_dataset(df_enrol, "enrol")
    agg_demo = aggregate_dataset(df_demo, "demo")
    agg_bio = aggregate_dataset(df_bio, "bio")
    
    # Merge all into one Master DataFrame
    # Outer join to ensure we keep districts even if they only appear in one dataset for a month
    master_df = agg_enrol.merge(agg_demo, on=['month', 'state', 'district'], how='outer')
    master_df = master_df.merge(agg_bio, on=['month', 'state', 'district'], how='outer')
    
    # Fill NaNs with 0 (assuming no record means 0 activity)
    master_df = master_df.fillna(0)
    
    return master_df

if __name__ == "__main__":
    # Quick test
    df = get_monthly_level_data()
    print("Master DF Shape:", df.shape)
    print(df.head())
    # Save a sample for inspection
    df.to_csv("src/processed_uidai_data_sample.csv", index=False)
