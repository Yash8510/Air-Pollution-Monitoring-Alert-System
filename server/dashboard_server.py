from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

sensor_data = {}
data_history = {}
MAX_HISTORY = 100

LOG_DIR = Path('sensor_logs')
LOG_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path('models')
MODEL_DIR.mkdir(exist_ok=True)

csv_headers = ['timestamp', 'temperature', 'humidity', 'pm25', 'pm10', 'co2', 'mq135_raw',
               'temperature_s', 'humidity_s', 'pm25_s', 'pm10_s', 'co2_s']

# AI/ML Models for each node
calibration_models = {}  # {node_id: {'model': Ridge(), 'scaler': StandardScaler(), 'active': bool}}
training_data = {}  # {node_id: {'X': [], 'y': []}}

# Anomaly detection parameters
anomaly_thresholds = {
    'temperature': {'min': -10, 'max': 60, 'std_multiplier': 3},
    'humidity': {'min': 0, 'max': 100, 'std_multiplier': 3},
    'pm25': {'min': 0, 'max': 500, 'std_multiplier': 4},
    'pm10': {'min': 0, 'max': 600, 'std_multiplier': 4},
    'co2': {'min': 300, 'max': 5000, 'std_multiplier': 3}
}


def get_node_csv_file(node_id: str) -> Path:
    log_date = datetime.now().strftime("%Y%m%d")
    return LOG_DIR / f"sensor_data_{node_id}_{log_date}.csv"


def init_csv_file(node_id: str) -> None:
    csv_path = get_node_csv_file(node_id)
    if csv_path.exists():
        return
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            writer.writeheader()
        print(f"Created CSV log file: {csv_path}")
    except Exception as e:
        print(f"Error creating CSV file for {node_id}: {e}")


def detect_anomalies(node_id: str, data: dict) -> dict:
    """Detect anomalies using statistical methods and return anomaly flags"""
    anomalies = {}
    
    if node_id not in data_history or len(data_history[node_id]) < 10:
        return {}  # Not enough data for anomaly detection
    
    history = data_history[node_id]
    
    for field in ['temperature', 'humidity', 'pm25', 'pm10', 'co2']:
        if field not in data:
            continue
            
        current_value = data[field]
        historical_values = [h[field] for h in history[-30:] if field in h]
        
        if len(historical_values) < 5:
            continue
        
        mean = np.mean(historical_values)
        std = np.std(historical_values)
        
        threshold = anomaly_thresholds.get(field, {})
        
        # Check hard limits
        if current_value < threshold.get('min', -1e9) or current_value > threshold.get('max', 1e9):
            anomalies[field] = 'out_of_range'
        # Check statistical deviation
        elif std > 0 and abs(current_value - mean) > threshold.get('std_multiplier', 3) * std:
            anomalies[field] = 'statistical_outlier'
    
    return anomalies


def apply_calibration_model(node_id: str, data: dict) -> dict:
    """Apply ML calibration model to sensor data if available"""
    if node_id not in calibration_models or not calibration_models[node_id].get('active', False):
        return {}
    
    try:
        model_info = calibration_models[node_id]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Build feature vector: [temp, humidity, pm25, pm10, mq135_raw]
        features = []
        for field in ['temperature', 'humidity', 'pm25', 'pm10', 'mq135_raw']:
            features.append(data.get(field, 0))
        
        X = np.array([features])
        X_scaled = scaler.transform(X)
        
        # Predict calibrated CO2
        co2_calibrated = model.predict(X_scaled)[0]
        
        return {
            'co2_calibrated': round(float(co2_calibrated), 2),
            'model_version': model_info.get('version', 1)
        }
    except Exception as e:
        print(f"Error applying calibration model for {node_id}: {e}")
        return {}


def get_model_coefficients(node_id: str) -> dict:
    """Get model coefficients to send to ESP8266 for on-device calibration"""
    if node_id not in calibration_models or not calibration_models[node_id].get('active', False):
        return {}
    
    try:
        model_info = calibration_models[node_id]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Return coefficients and scaling parameters
        return {
            'intercept': float(model.intercept_),
            'coefficients': model.coef_.tolist(),
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'feature_names': ['temperature', 'humidity', 'pm25', 'pm10', 'mq135_raw'],
            'version': model_info.get('version', 1)
        }
    except Exception as e:
        print(f"Error getting coefficients for {node_id}: {e}")
        return {}


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/api/sensor', methods=['POST'])
def receive_sensor():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON payload"}), 400
        
        node_id = data.get('node_id', 'UNKNOWN')
        
        if node_id not in sensor_data:
            sensor_data[node_id] = {}
            data_history[node_id] = []
            training_data[node_id] = {'X': [], 'y': []}
            init_csv_file(node_id)
        
        # Store raw sensor data
        sensor_entry = {
            'temperature': data.get('temperature', 0),
            'humidity': data.get('humidity', 0),
            'pm25': data.get('pm25', 0),
            'pm10': data.get('pm10', 0),
            'co2': data.get('co2', 0),
            'mq135_raw': data.get('mq135_raw', 0),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Add smoothed values if present
        if 'temperature_s' in data:
            sensor_entry['temperature_s'] = data.get('temperature_s', 0)
            sensor_entry['humidity_s'] = data.get('humidity_s', 0)
            sensor_entry['pm25_s'] = data.get('pm25_s', 0)
            sensor_entry['pm10_s'] = data.get('pm10_s', 0)
            sensor_entry['co2_s'] = data.get('co2_s', 0)
        
        # Anomaly detection
        anomalies = detect_anomalies(node_id, sensor_entry)
        if anomalies:
            sensor_entry['anomalies'] = anomalies
        
        # Apply ML calibration if available
        calibration = apply_calibration_model(node_id, sensor_entry)
        if calibration:
            sensor_entry['co2_calibrated'] = calibration['co2_calibrated']
            sensor_entry['model_version'] = calibration['model_version']
        
        sensor_data[node_id] = sensor_entry
        
        # Update history
        history_entry = sensor_entry.copy()
        data_history[node_id].append(history_entry)
        if len(data_history[node_id]) > MAX_HISTORY:
            data_history[node_id].pop(0)
        
        # Log to files
        log_to_csv(node_id, sensor_entry)
        log_to_json(node_id, sensor_entry)
        
        # Console output
        print(f"[{node_id}] {sensor_entry['timestamp']} | "
              f"Temp: {sensor_entry['temperature']:.1f}°C, "
              f"Humidity: {sensor_entry['humidity']:.1f}%, "
              f"PM2.5: {sensor_entry['pm25']:.1f}μg/m³, "
              f"PM10: {sensor_entry['pm10']:.1f}μg/m³, "
              f"CO2: {sensor_entry['co2']:.0f}ppm" +
              (f" (calibrated: {sensor_entry.get('co2_calibrated', 0):.0f}ppm)" if 'co2_calibrated' in sensor_entry else "") +
              (f" [ANOMALIES: {list(anomalies.keys())}]" if anomalies else ""))
        
        response = {"status": "success"}
        
        # Include calibration coefficients if model exists
        coeffs = get_model_coefficients(node_id)
        if coeffs:
            response['calibration'] = coeffs
        
        return jsonify(response), 200
    except Exception as e:
        print(f"Error receiving data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/api/sensor', methods=['GET'])
def get_sensor():
    return jsonify(sensor_data)


@app.route('/api/sensor/<node_id>', methods=['GET'])
def get_sensor_by_node(node_id):
    if node_id in sensor_data:
        return jsonify(sensor_data[node_id])
    return jsonify({"status": "error", "message": "Node not found"}), 404


@app.route('/api/sensor/history/<node_id>', methods=['GET'])
def get_history_by_node(node_id):
    if node_id in data_history:
        return jsonify(data_history[node_id])
    return jsonify({"status": "error", "message": "Node not found"}), 404


@app.route('/api/nodes/list', methods=['GET'])
def get_nodes_list():
    nodes = []
    for node_id, values in sensor_data.items():
        node_info = {
            'node_id': node_id,
            'last_update': values.get('timestamp', 'Never'),
            'temperature': values.get('temperature', 0),
            'humidity': values.get('humidity', 0),
            'pm25': values.get('pm25', 0),
            'pm10': values.get('pm10', 0),
            'co2': values.get('co2', 0)
        }
        
        # Add calibration status
        if node_id in calibration_models and calibration_models[node_id].get('active', False):
            node_info['calibrated'] = True
            node_info['co2_calibrated'] = values.get('co2_calibrated', 0)
        else:
            node_info['calibrated'] = False
        
        # Add anomaly flags
        if 'anomalies' in values:
            node_info['anomalies'] = values['anomalies']
        
        nodes.append(node_info)
    
    return jsonify({'nodes': nodes})


@app.route('/api/calibration/<node_id>/train', methods=['POST'])
def train_calibration_model(node_id):
    """Train calibration model using reference CO2 data
    
    Expects JSON: {
        "reference_co2": 450.0,  # Reference CO2 value from trusted sensor
        "auto_train": false      # If true, use last 50 samples for auto-calibration
    }
    """
    try:
        data = request.get_json()
        
        if node_id not in training_data:
            training_data[node_id] = {'X': [], 'y': []}
        
        if data.get('auto_train', False):
            # Auto-train using historical data (assumes outdoor clean air ~400-420 ppm)
            if node_id not in data_history or len(data_history[node_id]) < 20:
                return jsonify({
                    "status": "error",
                    "message": "Not enough historical data for auto-training (need 20+ samples)"
                }), 400
            
            # Use last 50 samples, assume outdoor baseline CO2
            history = data_history[node_id][-50:]
            baseline_co2 = 410  # Typical outdoor CO2
            
            for h in history:
                features = [
                    h.get('temperature', 0),
                    h.get('humidity', 0),
                    h.get('pm25', 0),
                    h.get('pm10', 0),
                    h.get('mq135_raw', 0)
                ]
                training_data[node_id]['X'].append(features)
                training_data[node_id]['y'].append(baseline_co2)
            
            print(f"Auto-training model for {node_id} with {len(history)} samples (baseline CO2: {baseline_co2})")
        
        else:
            # Manual training with reference CO2
            reference_co2 = data.get('reference_co2')
            if reference_co2 is None:
                return jsonify({
                    "status": "error",
                    "message": "reference_co2 value required for manual training"
                }), 400
            
            # Get current sensor reading
            if node_id not in sensor_data:
                return jsonify({
                    "status": "error",
                    "message": "No sensor data available for this node"
                }), 400
            
            current = sensor_data[node_id]
            features = [
                current.get('temperature', 0),
                current.get('humidity', 0),
                current.get('pm25', 0),
                current.get('pm10', 0),
                current.get('mq135_raw', 0)
            ]
            
            training_data[node_id]['X'].append(features)
            training_data[node_id]['y'].append(reference_co2)
            
            print(f"Added training sample for {node_id}: features={features}, target={reference_co2}")
        
        # Train model if we have enough samples
        n_samples = len(training_data[node_id]['X'])
        if n_samples >= 10:
            X = np.array(training_data[node_id]['X'])
            y = np.array(training_data[node_id]['y'])
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y)
            
            # Calculate R² score
            r2_score = model.score(X_scaled, y)
            
            # Save model
            calibration_models[node_id] = {
                'model': model,
                'scaler': scaler,
                'active': True,
                'version': calibration_models.get(node_id, {}).get('version', 0) + 1,
                'trained_at': datetime.now().isoformat(),
                'n_samples': n_samples,
                'r2_score': r2_score
            }
            
            # Save to disk
            model_file = MODEL_DIR / f"calibration_{node_id}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(calibration_models[node_id], f)
            
            print(f"✓ Model trained for {node_id}: {n_samples} samples, R²={r2_score:.3f}")
            
            return jsonify({
                "status": "success",
                "message": "Model trained successfully",
                "n_samples": n_samples,
                "r2_score": round(r2_score, 4),
                "version": calibration_models[node_id]['version'],
                "coefficients": get_model_coefficients(node_id)
            }), 200
        
        else:
            return jsonify({
                "status": "success",
                "message": f"Training sample added. Need {10 - n_samples} more samples to train model.",
                "n_samples": n_samples,
                "required": 10
            }), 200
    
    except Exception as e:
        print(f"Error training model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/calibration/<node_id>/status', methods=['GET'])
def get_calibration_status(node_id):
    """Get calibration model status and coefficients"""
    try:
        status = {
            'node_id': node_id,
            'calibrated': False,
            'n_training_samples': len(training_data.get(node_id, {}).get('X', []))
        }
        
        if node_id in calibration_models and calibration_models[node_id].get('active', False):
            model_info = calibration_models[node_id]
            status.update({
                'calibrated': True,
                'version': model_info.get('version', 1),
                'trained_at': model_info.get('trained_at'),
                'n_samples': model_info.get('n_samples', 0),
                'r2_score': round(model_info.get('r2_score', 0), 4),
                'coefficients': get_model_coefficients(node_id)
            })
        
        return jsonify(status), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/calibration/<node_id>/reset', methods=['POST'])
def reset_calibration(node_id):
    """Reset calibration model and training data"""
    try:
        if node_id in calibration_models:
            del calibration_models[node_id]
        
        if node_id in training_data:
            training_data[node_id] = {'X': [], 'y': []}
        
        # Delete model file
        model_file = MODEL_DIR / f"calibration_{node_id}.pkl"
        if model_file.exists():
            model_file.unlink()
        
        return jsonify({
            "status": "success",
            "message": "Calibration reset successfully"
        }), 200
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def log_to_csv(node_id, data):
    try:
        csv_path = get_node_csv_file(node_id)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_headers)
            row = {
                'timestamp': data['timestamp'],
                'temperature': round(data.get('temperature', 0), 2),
                'humidity': round(data.get('humidity', 0), 2),
                'pm25': round(data.get('pm25', 0), 2),
                'pm10': round(data.get('pm10', 0), 2),
                'co2': round(data.get('co2', 0), 2),
                'mq135_raw': data.get('mq135_raw', 0),
                'temperature_s': round(data.get('temperature_s', 0), 2),
                'humidity_s': round(data.get('humidity_s', 0), 2),
                'pm25_s': round(data.get('pm25_s', 0), 2),
                'pm10_s': round(data.get('pm10_s', 0), 2),
                'co2_s': round(data.get('co2_s', 0), 2)
            }
            writer.writerow(row)
    except Exception as e:
        print(f"Error writing to CSV for {node_id}: {e}")


def log_to_json(node_id, data):
    try:
        json_log_file = LOG_DIR / f"sensor_data_{node_id}_{datetime.now().strftime('%Y%m%d')}.jsonl"
        log_entry = {
            'timestamp': data['timestamp'],
            'temperature': round(data.get('temperature', 0), 2),
            'humidity': round(data.get('humidity', 0), 2),
            'pm25': round(data.get('pm25', 0), 2),
            'pm10': round(data.get('pm10', 0), 2),
            'co2': round(data.get('co2', 0), 2),
            'mq135_raw': data.get('mq135_raw', 0)
        }
        
        # Add smoothed values
        if 'temperature_s' in data:
            log_entry['temperature_s'] = round(data['temperature_s'], 2)
            log_entry['humidity_s'] = round(data['humidity_s'], 2)
            log_entry['pm25_s'] = round(data['pm25_s'], 2)
            log_entry['pm10_s'] = round(data['pm10_s'], 2)
            log_entry['co2_s'] = round(data['co2_s'], 2)
        
        # Add calibrated values
        if 'co2_calibrated' in data:
            log_entry['co2_calibrated'] = round(data['co2_calibrated'], 2)
        
        # Add anomaly flags
        if 'anomalies' in data:
            log_entry['anomalies'] = data['anomalies']
        
        with open(json_log_file, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')
    except Exception as e:
        print(f"Error writing to JSON for {node_id}: {e}")


@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        log_files = list(LOG_DIR.glob('*.csv'))
        return jsonify({
            'status': 'success',
            'log_files': [f.name for f in sorted(log_files, reverse=True)],
            'log_directory': str(LOG_DIR.absolute())
        }), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/logs/<filename>', methods=['GET'])
def download_log(filename):
    try:
        file_path = LOG_DIR / filename
        if not file_path.exists():
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        
        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    'timestamp': row['timestamp'],
                    'temperature': float(row.get('temperature', 0)),
                    'humidity': float(row.get('humidity', 0)),
                    'pm25': float(row.get('pm25', 0)),
                    'pm10': float(row.get('pm10', 0))
                }
                
                # Add optional fields if present
                if 'co2' in row and row['co2']:
                    entry['co2'] = float(row['co2'])
                if 'co2_s' in row and row['co2_s']:
                    entry['co2_s'] = float(row['co2_s'])
                
                data.append(entry)
        
        return jsonify({'status': 'success', 'data': data}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


def load_saved_models():
    """Load previously saved calibration models on startup"""
    model_files = list(MODEL_DIR.glob('calibration_*.pkl'))
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            node_id = model_file.stem.replace('calibration_', '')
            calibration_models[node_id] = model_data
            training_data[node_id] = {'X': [], 'y': []}
            
            print(f"✓ Loaded calibration model for {node_id} (v{model_data.get('version', 1)})")
        except Exception as e:
            print(f"Error loading model {model_file}: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  AIR QUALITY MONITORING SERVER WITH AI CALIBRATION")
    print("="*60)
    print(f"Sensor logs: {LOG_DIR.absolute()}")
    print(f"ML models: {MODEL_DIR.absolute()}")
    print("\nLoading saved calibration models...")
    load_saved_models()
    print("\n📡 Server starting on http://0.0.0.0:5000")
    print("\n🔧 API Endpoints:")
    print("  Data Collection:")
    print("    POST /api/sensor - Receive sensor data")
    print("    GET  /api/sensor - Get all current data")
    print("    GET  /api/sensor/<node_id> - Get node data")
    print("    GET  /api/sensor/history/<node_id> - Get history")
    print("    GET  /api/nodes/list - List all nodes")
    print("\n  🤖 AI Calibration:")
    print("    POST /api/calibration/<node_id>/train - Train calibration model")
    print("         Body: {\"reference_co2\": 450} OR {\"auto_train\": true}")
    print("    GET  /api/calibration/<node_id>/status - Get model status")
    print("    POST /api/calibration/<node_id>/reset - Reset calibration")
    print("\n  📊 Data Logs:")
    print("    GET  /api/logs - List log files")
    print("    GET  /api/logs/<filename> - Download log")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
