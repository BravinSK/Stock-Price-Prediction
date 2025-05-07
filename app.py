from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from werkzeug.serving import WSGIRequestHandler
import os
import json
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
from pathlib import Path

from config import PORT, HOST, DEBUG
from model import train_model, ensure_model_exists
from predict import predict_future_prices, plot_prediction_results
from utils import create_directory_if_not_exists

# Increase Werkzeug server timeout
WSGIRequestHandler.protocol_version = "HTTP/1.1"

# Create necessary directories
create_directory_if_not_exists('static/images')
create_directory_if_not_exists('data')
create_directory_if_not_exists('models')

# Initialize model if needed
ensure_model_exists()

# Add constants for model files
MODEL_PATH = os.path.join('models', 'pretrained_model.h5')
SCALER_PATH = os.path.join('models', 'scaler.pkl')
IS_PRETRAINED = Path(MODEL_PATH).exists() and Path(SCALER_PATH).exists()

# Create Flask app with proper CORS configuration
app = Flask(__name__)
CORS(app)

# Configure app
app.config.update(
    SEND_FILE_MAX_AGE_DEFAULT=0,
    CORS_HEADERS='Content-Type',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    PROPAGATE_EXCEPTIONS=True
)

# Add timeout for training status
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'error': None,
    'last_updated': None
}
status_lock = threading.Lock()

def update_training_status(progress, message, error=None):
    global training_status
    with status_lock:
        training_status['progress'] = progress
        training_status['message'] = message
        training_status['error'] = error
        training_status['last_updated'] = datetime.now()

def reset_stale_training():
    """Reset training status if it's stale (no updates for 5 minutes)"""
    global training_status
    with status_lock:
        if not training_status['is_training']:
            return False
        
        if training_status['last_updated'] is None:
            training_status['is_training'] = False
            return True
            
        time_diff = (datetime.now() - training_status['last_updated']).total_seconds()
        if time_diff > 300:  # 5 minutes
            training_status['is_training'] = False
            training_status['error'] = 'Training process timed out'
            training_status['progress'] = 0
            return True
    return False

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/prediction/<timestamp>')
def prediction_view(timestamp):
    """Render the prediction results page"""
    # Load prediction data from session or store
    # This is a simplified version; in a real app, you'd store prediction data
    try:
        # For demonstration, we'll just pass empty data
        # In a real application, you would store prediction results in a database or session
        prediction_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_days': 5,
            'plot_path': f'images/prediction_{timestamp}.png',
            'input_data': [
                {'date': '2025-05-01', 'close': 800.00},
                {'date': '2025-05-02', 'close': 805.00},
                # ... more data would be here in a real app
            ],
            'predictions': {
                'dates': ['2025-05-06', '2025-05-07', '2025-05-08', '2025-05-09', '2025-05-10'],
                'predictions': [810.00, 815.00, 820.00, 825.00, 830.00]
            }
        }
        
        return render_template('prediction.html', prediction_data=prediction_data)
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/train', methods=['POST', 'OPTIONS'])
def train():
    """Start the training process"""
    if request.method == 'OPTIONS':
        return handle_preflight()

    global training_status
    with status_lock:
        # Check if we already have a pre-trained model
        if IS_PRETRAINED:
            return jsonify({
                'status': 'success',
                'message': 'Model is already trained and ready for predictions.',
                'pretrained': True
            })

        # Check for data file
        if not os.path.exists(os.path.join('data', 'Tesla.csv')):
            return jsonify({
                'status': 'error',
                'message': 'Please upload the Tesla.csv file first.'
            }), 400

        if training_status['is_training']:
            return jsonify({
                'status': 'info',
                'message': 'Training already in progress',
                'progress': training_status['progress']
            }), 409

        training_status['is_training'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Starting training...'
        training_status['error'] = None
        training_status['last_updated'] = datetime.now()

    def train_in_thread():
        try:
            update_training_status(10, 'Loading data...')
            # Train the model using Tesla.csv
            model, history, metrics = train_model(data_path='data/Tesla.csv',
                                                model_save_path=MODEL_PATH,
                                                scaler_save_path=SCALER_PATH)
            
            if model is not None and history is not None:
                update_training_status(100, 'Training completed successfully!')
            else:
                update_training_status(0, 'Training failed', 'Model training failed')
        except Exception as e:
            update_training_status(0, 'Training failed', str(e))
        finally:
            with status_lock:
                training_status['is_training'] = False

    thread = threading.Thread(target=train_in_thread)
    thread.daemon = True
    thread.start()

    return jsonify({
        'status': 'success',
        'message': 'Training started',
        'progress': 0
    })

@app.route('/train/status', methods=['GET'])
def get_training_status():
    """Get the current training status"""
    global training_status
    with status_lock:
        return jsonify({
            'is_training': training_status['is_training'],
            'progress': training_status['progress'],
            'message': training_status['message'],
            'error': training_status['error']
        })

@app.route('/model/status', methods=['GET'])
def get_model_status():
    """Check if model is pre-trained"""
    return jsonify({
        'is_trained': IS_PRETRAINED,
        'message': 'Model is ready' if IS_PRETRAINED else 'Model needs training'
    })

def handle_preflight():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload CSV file"""
    if 'file' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No file part in the request'
        })
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        })
    
    if file and file.filename.endswith('.csv'):
        try:
            # Save the file
            file_path = os.path.join('data', 'Tesla.csv')
            file.save(file_path)
            
            # Read the file to verify
            df = pd.read_csv(file_path)
            
            return jsonify({
                'status': 'success',
                'message': 'File uploaded successfully',
                'rows': len(df),
                'columns': df.columns.tolist()
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error processing file: {str(e)}'
            })
    else:
        return jsonify({
            'status': 'error',
            'message': 'Only CSV files are allowed'
        })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    try:
        # Get data from request
        data = request.get_json()  # Changed from request.json
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data received'
            }), 400  # Added status code
        
        input_data = data.get('input_data', [])
        num_days = int(data.get('num_days', 1))
        
        if not input_data or num_days < 1:
            return jsonify({
                'status': 'error',
                'message': 'Invalid input data or number of days'
            }), 400  # Added status code
        
        # Make predictions
        result = predict_future_prices(input_data, num_days)
        
        if 'error' in result:
            return jsonify({
                'status': 'error',
                'message': result['error']
            }), 500  # Added status code
        
        # Generate plot
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        plot_path = f'images/prediction_{timestamp}.png'
        plot_prediction_results(input_data, result, save_path=f'static/{plot_path}')
        
        # Return the result with plot path
        return jsonify({
            'status': 'success',
            'predictions': result,
            'plot_path': plot_path
        }), 200  # Added status code
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error making prediction: {str(e)}'
        }), 500  # Added status code

if __name__ == '__main__':
    app.run(
        host=HOST, 
        port=PORT, 
        debug=DEBUG, 
        threaded=True,
        processes=1,
        request_handler=WSGIRequestHandler
    )