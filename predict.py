import numpy as np
from tensorflow.keras.models import load_model
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from model import ensure_model_exists

def predict_future_prices(input_data, num_days):
    """Predict future stock prices"""
    try:
        # Ensure model exists
        if not ensure_model_exists():
            return {'error': 'Model not available. Please upload data and train the model first.'}
            
        # Load the trained model and scaler
        model = load_model('models/pretrained_model.h5')
        scaler = joblib.load('models/scaler.pkl')
        
        # Extract close prices and convert to numpy array
        close_prices = np.array([x['close'] for x in input_data]).reshape(-1, 1)
        
        # Scale the input data
        scaled_data = scaler.transform(close_prices)
        
        # Prepare input sequence (last 10 days)
        sequence = scaled_data[-10:]  # Get last 10 days
        sequence = sequence.reshape(1, 10, 1)  # Reshape to (1, timesteps, features)
        
        # Make predictions
        predictions = []
        dates = []
        last_sequence = sequence.copy()
        
        # Generate future dates
        last_date = datetime.strptime(input_data[-1]['date'], '%Y-%m-%d')
        
        for i in range(num_days):
            # Predict next value
            next_pred = model.predict(last_sequence, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence[0], -1)
            last_sequence[-1] = next_pred
            last_sequence = last_sequence.reshape(1, 10, 1)
            
            # Generate next date
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date.strftime('%Y-%m-%d'))
        
        # Inverse transform predictions to get actual prices
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions)
        
        return {
            'dates': dates,
            'predictions': predictions.flatten().tolist()
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {'error': str(e)}

def plot_prediction_results(input_data, predictions, save_path):
    """Plot the input data and predictions"""
    try:
        # Prepare data for plotting
        input_dates = [x['date'] for x in input_data]
        input_prices = [x['close'] for x in input_data]
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        # Plot input data
        plt.plot(input_dates, input_prices, 'b-', label='Historical Data', marker='o')
        
        # Plot predictions
        plt.plot(predictions['dates'], predictions['predictions'], 'r--', label='Predictions', marker='o')
        
        # Customize the plot
        plt.title('Tesla Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price ($)')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Plot error: {str(e)}")
        return None