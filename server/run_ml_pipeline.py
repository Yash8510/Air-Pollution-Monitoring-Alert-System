#!/usr/bin/env python3
"""
Quick Start Script for ML Model Pipeline
Simple one-command execution for full ML workflow
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and display progress"""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    """Execute full ML pipeline"""
    
    print("\n" + "="*60)
    print("AIR QUALITY ML MODEL - QUICK START")
    print("="*60)
    
    server_dir = Path(__file__).parent
    
    # Check if synthetic data exists
    synthetic_data = server_dir / 'sensor_logs' / 'synthetic_sensor_data.csv'
    
    if not synthetic_data.exists():
        print("\n📊 Step 1: Generating synthetic training data...")
        if not run_command(f'python generate_synthetic_data.py', 
                          'Generate Synthetic Data (24 hours)'):
            print("❌ Failed to generate synthetic data")
            sys.exit(1)
    else:
        print("\n✓ Synthetic data already exists")
    
    # Build ML models
    print("\n🤖 Step 2: Training ML models...")
    if not run_command(f'python ml_model.py', 
                      'Train Models & Make Predictions'):
        print("❌ Failed to train models")
        sys.exit(1)
    
    # Generate visualizations
    print("\n📈 Step 3: Generating visualizations...")
    if not run_command(f'python visualize_data.py', 
                      'Generate Analysis Plots'):
        print("❌ Failed to generate visualizations")
        sys.exit(1)
    
    # Success summary
    print("\n" + "="*60)
    print("✓ ML PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   • models/random_forest_model.pkl")
    print("   • models/gradient_boosting_model.pkl")
    print("   • models/linear_regression_model.pkl")
    print("   • analysis_output/time_series.png")
    print("   • analysis_output/correlations.png")
    print("   • analysis_output/hourly_patterns.png")
    print("   • analysis_output/aqi_distribution.png")
    print("   • analysis_output/anomaly_detection.png")
    
    print("\n📚 Next Steps:")
    print("   1. View ML_README.md for detailed documentation")
    print("   2. Check analysis_output/ for visualization results")
    print("   3. Integrate models into dashboard_server.py")
    print("   4. Collect real sensor data to improve model")
    
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()
