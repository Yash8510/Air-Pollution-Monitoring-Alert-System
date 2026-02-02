import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_all_sensor_data(data_dir='sensor_logs'):
    csv_files = list(Path(data_dir).glob('sensor_data_*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for csv_file in sorted(csv_files):
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dfs.append(df)
        print(f"Loaded: {csv_file.name} ({len(df)} records)")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nCombined dataset: {len(combined)} unique records")
    return combined

def calculate_aqi_from_pm25(pm25):
    if pm25 <= 12.0:
        return (pm25 / 12.0) * 50
    elif pm25 <= 35.4:
        return 50 + ((pm25 - 12.1) / (35.4 - 12.1)) * 50
    elif pm25 <= 55.4:
        return 100 + ((pm25 - 35.5) / (55.4 - 35.5)) * 50
    elif pm25 <= 150.4:
        return 150 + ((pm25 - 55.5) / (150.4 - 55.5)) * 50
    elif pm25 <= 250.4:
        return 200 + ((pm25 - 150.5) / (250.4 - 150.5)) * 100
    else:
        return 300 + ((pm25 - 250.5) / 50.0) * 50

def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def get_aqi_color(aqi_value):
    category = get_aqi_category(aqi_value)
    colors = {
        "Good": (0, 200, 0),
        "Moderate": (200, 200, 0),
        "Unhealthy for Sensitive Groups": (255, 165, 0),
        "Unhealthy": (255, 0, 0),
        "Very Unhealthy": (128, 0, 128),
        "Hazardous": (139, 0, 0)
    }
    return colors.get(category, (0, 0, 0))

def detect_anomalies(data, column, window=5, threshold=2.5):
    rolling_mean = data[column].rolling(window=window).mean()
    rolling_std = data[column].rolling(window=window).std()
    
    z_score = np.abs((data[column] - rolling_mean) / rolling_std)
    anomalies = z_score > threshold
    
    return anomalies, z_score

def resample_data(df, freq='5min'):
    df_resampled = df.set_index('timestamp').resample(freq).mean()
    df_resampled = df_resampled.reset_index()
    return df_resampled

def data_quality_report(df):
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    print(f"\nTotal Records: {len(df)}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600:.2f} hours")
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  None")
    
    print("\nStatistical Summary:")
    numeric_cols = ['temperature', 'humidity', 'pm25', 'pm10', 'co2']
    for col in numeric_cols:
        if col in df.columns:
            print(f"\n  {col.upper()}:")
            print(f"    Mean: {df[col].mean():.2f}")
            print(f"    Std:  {df[col].std():.2f}")
            print(f"    Min:  {df[col].min():.2f}")
            print(f"    Max:  {df[col].max():.2f}")
    
    return missing


if __name__ == '__main__':
    data = load_all_sensor_data()
    data_quality_report(data)
