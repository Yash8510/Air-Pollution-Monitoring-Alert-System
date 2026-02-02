"""
Visualization and detailed analysis of sensor data and model predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from ml_utils import load_all_sensor_data, calculate_aqi_from_pm25, get_aqi_category, detect_anomalies

class DataVisualization:
    """Handle data visualization and analysis plotting"""
    
    def __init__(self, data_dir='sensor_logs'):
        self.data_dir = data_dir
        self.df = None
        self.output_dir = Path('analysis_output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 6)
    
    def load_data(self):
        """Load sensor data"""
        self.df = load_all_sensor_data(self.data_dir)
        self.df['aqi'] = self.df['pm25'].apply(calculate_aqi_from_pm25)
        self.df['aqi_category'] = self.df['aqi'].apply(get_aqi_category)
        return self.df
    
    def plot_time_series(self):
        """Plot all sensor readings over time"""
        print("Generating time series plot...")
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # PM levels
        axes[0].plot(self.df['timestamp'], self.df['pm25'], label='PM2.5', marker='o', markersize=3)
        axes[0].plot(self.df['timestamp'], self.df['pm10'], label='PM10', marker='s', markersize=3)
        axes[0].set_ylabel('PM Concentration (μg/m³)')
        axes[0].set_title('Particulate Matter Levels Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Temperature & Humidity
        ax1 = axes[1]
        ax2 = ax1.twinx()
        
        ax1.plot(self.df['timestamp'], self.df['temperature'], 'r-', label='Temperature', marker='o', markersize=3)
        ax2.plot(self.df['timestamp'], self.df['humidity'], 'b-', label='Humidity', marker='s', markersize=3)
        
        ax1.set_ylabel('Temperature (°C)', color='r')
        ax2.set_ylabel('Humidity (%)', color='b')
        ax1.set_title('Temperature & Humidity Over Time')
        ax1.tick_params(axis='y', labelcolor='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)
        
        # CO2 and AQI
        ax3 = axes[2]
        ax4 = ax3.twinx()
        
        ax3.plot(self.df['timestamp'], self.df['co2'], 'g-', label='CO2 (est)', marker='o', markersize=3)
        ax4.plot(self.df['timestamp'], self.df['aqi'], 'r--', label='AQI', marker='D', markersize=3, linewidth=2)
        
        ax3.set_ylabel('CO2 (ppm)', color='g')
        ax4.set_ylabel('AQI Index', color='r')
        ax3.set_xlabel('Time')
        ax3.set_title('CO2 Estimation & AQI Over Time')
        ax3.tick_params(axis='y', labelcolor='g')
        ax4.tick_params(axis='y', labelcolor='r')
        ax3.grid(True, alpha=0.3)
        
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        filepath = self.output_dir / 'time_series.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()
    
    def plot_correlations(self):
        """Plot correlation heatmap"""
        print("Generating correlation heatmap...")
        
        numeric_cols = ['temperature', 'humidity', 'pm25', 'pm10', 'co2', 'aqi']
        corr_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                    square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = self.output_dir / 'correlations.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()
    
    def plot_hourly_patterns(self):
        """Plot hourly AQI patterns"""
        print("Generating hourly pattern plot...")
        
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly_stats = self.df.groupby('hour').agg({
            'aqi': ['mean', 'std', 'min', 'max'],
            'pm25': 'mean',
            'pm10': 'mean'
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # AQI pattern
        hour = hourly_stats['hour'].values
        aqi_mean = hourly_stats[('aqi', 'mean')].values
        aqi_std = hourly_stats[('aqi', 'std')].values
        
        axes[0].plot(hour, aqi_mean, 'b-', linewidth=2, marker='o', label='Mean AQI')
        axes[0].fill_between(hour, aqi_mean - aqi_std, aqi_mean + aqi_std, alpha=0.3)
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('AQI')
        axes[0].set_title('Average AQI Pattern by Hour')
        axes[0].set_xticks(range(0, 24, 2))
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # PM pattern
        axes[1].plot(hour, hourly_stats[('pm25', 'mean')].values, 'r-', linewidth=2, 
                    marker='o', label='PM2.5')
        axes[1].plot(hour, hourly_stats[('pm10', 'mean')].values, 'orange', linewidth=2, 
                    marker='s', label='PM10')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('PM Concentration (μg/m³)')
        axes[1].set_title('Average PM Pattern by Hour')
        axes[1].set_xticks(range(0, 24, 2))
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        filepath = self.output_dir / 'hourly_patterns.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()
    
    def plot_aqi_distribution(self):
        """Plot AQI distribution and categories"""
        print("Generating AQI distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(self.df['aqi'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('AQI Value')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('AQI Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Category pie chart
        category_counts = self.df['aqi_category'].value_counts()
        colors_map = {
            'Good': '#00C800',
            'Moderate': '#C8C800',
            'Unhealthy for Sensitive Groups': '#FFA500',
            'Unhealthy': '#FF0000',
            'Very Unhealthy': '#800080',
            'Hazardous': '#8B0000'
        }
        
        colors = [colors_map.get(cat, '#000000') for cat in category_counts.index]
        axes[1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
        axes[1].set_title('AQI Category Distribution')
        
        plt.tight_layout()
        filepath = self.output_dir / 'aqi_distribution.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()
    
    def plot_anomaly_detection(self):
        """Plot detected anomalies"""
        print("Generating anomaly detection plot...")
        
        anomalies, z_scores = detect_anomalies(self.df, 'pm25', window=5, threshold=2.5)
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        ax.plot(self.df['timestamp'], self.df['pm25'], 'b-', label='PM2.5', alpha=0.7)
        
        # Highlight anomalies
        anomaly_points = self.df[anomalies]
        ax.scatter(anomaly_points['timestamp'], anomaly_points['pm25'], 
                  color='red', s=100, marker='X', label=f'Anomalies (n={anomalies.sum()})', zorder=5)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('PM2.5 (μg/m³)')
        ax.set_title('Anomaly Detection in PM2.5 Levels (Z-score > 2.5)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        filepath = self.output_dir / 'anomaly_detection.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filepath}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        self.load_data()
        self.plot_time_series()
        self.plot_correlations()
        self.plot_hourly_patterns()
        self.plot_aqi_distribution()
        self.plot_anomaly_detection()
        
        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    """Main execution"""
    visualizer = DataVisualization()
    visualizer.generate_all_plots()


if __name__ == '__main__':
    main()
