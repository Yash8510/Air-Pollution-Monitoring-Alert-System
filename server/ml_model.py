"""
Air Quality Pattern Detection and Prediction Model
Uses historical sensor data to identify patterns and predict future AQI levels
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from pathlib import Path

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class AQIPredictor:
    """Main class for AQI pattern detection and prediction"""
    
    def __init__(self, data_dir='sensor_logs'):
        self.data_dir = data_dir
        self.df = None
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importance = {}
        
        # Model storage
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
    
    def load_data(self, filename=None):
        """Load sensor data from CSV"""
        if filename is None:
            # Prefer synthetic data, fallback to real data
            csv_files = list(Path(self.data_dir).glob('*synthetic*.csv'))
            if not csv_files:
                csv_files = list(Path(self.data_dir).glob('sensor_data_*.csv'))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
            filename = sorted(csv_files)[-1]
        
        filepath = Path(self.data_dir) / filename if not Path(filename).exists() else filename
        
        print(f"Loading data from: {filepath}")
        self.df = pd.read_csv(filepath)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Calculate AQI from PM2.5 (EPA standard)
        self.df['aqi'] = self.calculate_aqi(self.df['pm25'].values)
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Loaded {len(self.df)} records")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"\nData Summary:\n{self.df.describe()}")
        
        return self.df
    
    @staticmethod
    def calculate_aqi(pm25_values):
        """Calculate AQI from PM2.5 using EPA standard"""
        aqi = np.zeros_like(pm25_values, dtype=float)
        
        for i, pm in enumerate(pm25_values):
            if pm <= 12.0:
                aqi[i] = (pm * 50.0 / 12.0)
            elif pm <= 35.4:
                aqi[i] = (50.0 + (pm - 12.1) * 50.0 / 23.3)
            elif pm <= 55.4:
                aqi[i] = (100.0 + (pm - 35.5) * 50.0 / 19.9)
            elif pm <= 150.4:
                aqi[i] = (150.0 + (pm - 55.5) * 50.0 / 94.9)
            elif pm <= 250.4:
                aqi[i] = (200.0 + (pm - 150.5) * 100.0 / 99.9)
            else:
                aqi[i] = (300.0 + (pm - 250.5) * 50.0 / 50.0)
        
        return np.minimum(aqi, 500)
    
    def create_features(self, lookback=5):
        """Create time-series features from raw data"""
        print(f"\nCreating features with lookback period of {lookback} samples...")
        
        # Extract time-based features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['dayofweek'] = self.df['timestamp'].dt.dayofweek
        self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
        
        # Create lagged features for time-series
        for lag in range(1, lookback + 1):
            self.df[f'pm25_lag_{lag}'] = self.df['pm25'].shift(lag)
            self.df[f'aqi_lag_{lag}'] = self.df['aqi'].shift(lag)
            self.df[f'temp_lag_{lag}'] = self.df['temperature'].shift(lag)
        
        # Calculate rolling statistics
        for window in [3, 5]:
            self.df[f'pm25_rolling_mean_{window}'] = self.df['pm25'].rolling(window=window).mean()
            self.df[f'aqi_rolling_mean_{window}'] = self.df['aqi'].rolling(window=window).mean()
        
        # Drop NaN rows created by lag features
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Features created. Shape: {self.df.shape}")
        return self.df
    
    def detect_patterns(self):
        """Detect patterns in the data"""
        print("\n" + "="*60)
        print("PATTERN ANALYSIS")
        print("="*60)
        
        # Extract time features if not already present
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df['timestamp'].dt.hour
        if 'dayofweek' not in self.df.columns:
            self.df['dayofweek'] = self.df['timestamp'].dt.dayofweek
        if 'is_weekend' not in self.df.columns:
            self.df['is_weekend'] = (self.df['dayofweek'] >= 5).astype(int)
        
        # Hourly pattern
        if len(self.df['hour'].unique()) > 1:
            hourly_aqi = self.df.groupby('hour')['aqi'].agg(['mean', 'std', 'count'])
            print("\nHourly AQI Pattern:")
            print(hourly_aqi)
        else:
            print("\nHourly AQI Pattern: (Insufficient data - only 1 hour represented)")
            print(f"  Hour {self.df['hour'].iloc[0]}: mean AQI = {self.df['aqi'].mean():.2f}")
        
        # Weekend vs Weekday
        if len(self.df['is_weekend'].unique()) > 1:
            weekend_aqi = self.df.groupby('is_weekend')['aqi'].agg(['mean', 'std'])
            weekend_aqi.index = ['Weekday', 'Weekend']
            print("\nWeekend vs Weekday AQI:")
            print(weekend_aqi)
        else:
            print("\nWeekend vs Weekday AQI: (Insufficient data - only one day type)")
        
        # Correlation analysis
        numeric_cols = ['temperature', 'humidity', 'pm25', 'pm10', 'co2', 'aqi']
        correlation = self.df[numeric_cols].corr()
        print("\nFeature Correlation with AQI:")
        print(correlation['aqi'].sort_values(ascending=False))
        
        # Pollution levels by time of day
        if len(self.df['hour'].unique()) > 1:
            self.df['time_period'] = pd.cut(self.df['hour'], 
                                            bins=[0, 6, 12, 18, 24],
                                            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                                            include_lowest=True)
            time_aqi = self.df.groupby('time_period')['aqi'].agg(['mean', 'min', 'max', 'std'])
            print("\nAQI by Time Period:")
            print(time_aqi)
        else:
            print("\nAQI by Time Period: (Insufficient data)")
        
        return {
            'hourly': None,
            'weekend_vs_weekday': None,
            'correlation': correlation,
            'time_period': None
        }
    
    def build_models(self, test_size=0.2, random_state=42):
        """Build multiple prediction models"""
        print("\n" + "="*60)
        print("BUILDING PREDICTION MODELS")
        print("="*60)
        
        # Check if we have enough data
        if len(self.df) < 20:
            print(f"\n⚠️  WARNING: Only {len(self.df)} records available.")
            print("   Recommended minimum: 100+ records for reliable models")
            print("   Continue anyway: Using 80/20 split (even with small dataset)")
        
        # Prepare features and target
        feature_cols = [col for col in self.df.columns 
                       if col not in ['timestamp', 'aqi', 'time_period', 'temperature_s', 'humidity_s', 
                                     'pm25_s', 'pm10_s', 'co2_s']]
        
        X = self.df[feature_cols].fillna(0)
        y = self.df['aqi']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Adjust test size for small datasets
        actual_test_size = max(0.2, min(1 - 2/len(self.df), test_size))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=actual_test_size, random_state=random_state
        )
        
        print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Model 1: Random Forest (Best for feature importance)
        print("\n[1/3] Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=random_state, n_jobs=-1)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        print(f"  R² Score: {rf_r2:.4f}, MAE: {rf_mae:.2f}")
        self.models['random_forest'] = rf_model
        self.feature_importance['random_forest'] = dict(zip(feature_cols, rf_model.feature_importances_))
        
        # Model 2: Gradient Boosting (Often more accurate)
        print("\n[2/3] Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=random_state)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_r2 = r2_score(y_test, gb_pred)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        print(f"  R² Score: {gb_r2:.4f}, MAE: {gb_mae:.2f}")
        self.models['gradient_boosting'] = gb_model
        self.feature_importance['gradient_boosting'] = dict(zip(feature_cols, gb_model.feature_importances_))
        
        # Model 3: Linear Regression (Baseline)
        print("\n[3/3] Training Linear Regression (Baseline)...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_r2 = r2_score(y_test, lr_pred)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        print(f"  R² Score: {lr_r2:.4f}, MAE: {lr_mae:.2f}")
        self.models['linear_regression'] = lr_model
        
        # Best model summary
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        results = {
            'Random Forest': {'R2': rf_r2, 'MAE': rf_mae},
            'Gradient Boosting': {'R2': gb_r2, 'MAE': gb_mae},
            'Linear Regression': {'R2': lr_r2, 'MAE': lr_mae}
        }
        
        for model_name, metrics in results.items():
            print(f"{model_name:20s} - R²: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.2f}")
        
        best_model = max(results.items(), key=lambda x: x[1]['R2'])
        print(f"\nBest Model: {best_model[0]} (R² = {best_model[1]['R2']:.4f})")
        
        # Feature importance
        if len(X) >= 5:
            print("\n" + "="*60)
            print("TOP 10 IMPORTANT FEATURES (Random Forest)")
            print("="*60)
            importance_df = pd.DataFrame(
                list(self.feature_importance['random_forest'].items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            print(importance_df.head(10).to_string(index=False))
        
        return results
    
    def predict_future(self, hours_ahead=6):
        """Predict AQI for future hours"""
        print(f"\n" + "="*60)
        print(f"PREDICTING AQI FOR NEXT {hours_ahead} HOURS")
        print("="*60)
        
        # Get last record and prepare for prediction
        last_record = self.df.iloc[-1].copy()
        last_time = last_record['timestamp']
        
        feature_cols = [col for col in self.df.columns 
                       if col not in ['timestamp', 'aqi', 'time_period', 'temperature_s', 'humidity_s', 
                                     'pm25_s', 'pm10_s', 'co2_s']]
        
        best_model = self.models.get('gradient_boosting') or self.models.get('random_forest')
        
        predictions = []
        current_record = self.df.iloc[-1].copy()
        
        for hour_ahead in range(1, hours_ahead + 1):
            future_time = last_time + timedelta(hours=hour_ahead)
            
            # Create feature vector for prediction
            X_future = current_record[feature_cols].values.reshape(1, -1)
            X_future_scaled = self.scaler.transform(X_future)
            
            pred_aqi = best_model.predict(X_future_scaled)[0]
            pred_aqi = max(0, min(500, pred_aqi))  # Clamp between 0-500
            
            # Determine air quality category
            if pred_aqi <= 50:
                category = "Good"
            elif pred_aqi <= 100:
                category = "Moderate"
            elif pred_aqi <= 150:
                category = "Unhealthy for Sensitive"
            elif pred_aqi <= 200:
                category = "Unhealthy"
            elif pred_aqi <= 300:
                category = "Very Unhealthy"
            else:
                category = "Hazardous"
            
            predictions.append({
                'time': future_time,
                'hours_ahead': hour_ahead,
                'predicted_aqi': round(pred_aqi, 2),
                'category': category
            })
            
            print(f"[+{hour_ahead}h] {future_time.strftime('%H:%M')} → AQI: {pred_aqi:.2f} ({category})")
        
        return pd.DataFrame(predictions)
    
    def save_models(self):
        """Save trained models to disk"""
        print("\nSaving models...")
        for name, model in self.models.items():
            filepath = self.model_dir / f'{name}_model.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved: {filepath}")
        
        # Save scaler
        with open(self.model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  Saved: scaler.pkl")
    
    def load_models(self):
        """Load trained models from disk"""
        print("Loading models...")
        for model_file in self.model_dir.glob('*_model.pkl'):
            model_name = model_file.stem.replace('_model', '')
            with open(model_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
            print(f"  Loaded: {model_name}")
        
        # Load scaler
        scaler_file = self.model_dir / 'scaler.pkl'
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)


def main():
    """Main execution"""
    
    # Initialize predictor
    predictor = AQIPredictor(data_dir='sensor_logs')
    
    # Load and explore data
    predictor.load_data()
    
    # Detect patterns
    patterns = predictor.detect_patterns()
    
    # Create features
    predictor.create_features(lookback=5)
    
    # Build models
    predictor.build_models()
    
    # Make predictions
    future_predictions = predictor.predict_future(hours_ahead=6)
    
    # Save models
    predictor.save_models()
    
    print("\n" + "="*60)
    print("✓ ML Model Pipeline Complete")
    print("="*60)


if __name__ == '__main__':
    main()
