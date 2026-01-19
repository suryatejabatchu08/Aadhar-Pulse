import pandas as pd
import numpy as np
from prophet import Prophet
import logging
import warnings

# Suppress Prophet warnings
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

def train_and_forecast(df, district_name, periods=3):
    """
    Trains a Prophet model on the 'stress_score' of a specific district
    and forecasts 'periods' months into the future.
    """
    # Filter for district
    district_df = df[df['district'] == district_name].copy()
    
    if len(district_df) < 2:
        return None  # Not enough data
    
    # Prepare for Prophet: ds (date), y (value)
    # Ensure month is datetime
    if not np.issubdtype(district_df['month'].dtype, np.datetime64):
         district_df['month'] = pd.to_datetime(district_df['month'])
         
    prophet_df = district_df[['month', 'stress_score']].rename(columns={'month': 'ds', 'stress_score': 'y'})
    
    # Train
    model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    
    # Forecast
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def detect_anomalies(df):
    """
    Detects anomalies in the stress score using simple statistical thresholding (Mean + 2*StdDev).
    Returns the dataframe with an 'is_anomaly' flag.
    """
    if df.empty:
        return df

    # Calculate stats per district
    # We can use transform to keep the original shape
    district_groups = df.groupby('district')['stress_score']
    means = district_groups.transform('mean')
    stds = district_groups.transform('std')
    
    # Z-score
    df['z_score'] = (df['stress_score'] - means) / (stds.replace(0, 1)) # avoid div by zero
    
    # Flag > 2 sigma
    df['is_anomaly'] = df['z_score'].abs() > 2.0
    
    return df

if __name__ == "__main__":
    # Test
    try:
        df = pd.read_csv("src/scored_uidai_data.csv")
        df['month'] = pd.to_datetime(df['month'])
        
        # Anomaly Detection
        df = detect_anomalies(df)
        print("Anomalies Detected:", df['is_anomaly'].sum())
        
        # Forecast sample district
        sample_district = df['district'].iloc[0]
        print(f"Forecasting for {sample_district}...")
        forecast = train_and_forecast(df, sample_district)
        if forecast is not None:
             print(forecast)
        
        # Save output with anomalies
        df.to_csv("src/analyzed_uidai_data.csv", index=False)
        print("Saved analyzed data.")
        
    except Exception as e:
        print(f"Error in models: {e}")
