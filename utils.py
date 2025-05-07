import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from config import MODEL_PATH, SCALER_PATH
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_directory_if_not_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_model_and_scaler():
    """Load the trained LSTM model and scaler"""
    from tensorflow.keras.models import load_model
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    return model, scaler

def generate_future_dates(start_date, n_days):
    """Generate future dates starting from start_date for n_days"""
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    dates = [start_date + timedelta(days=i) for i in range(1, n_days + 1)]
    return [date.strftime('%Y-%m-%d') for date in dates]

def plot_stock_prediction(actual_dates, actual_prices, pred_dates, pred_prices):
    """
    Create a plot of actual vs predicted stock prices
    
    Parameters:
    -----------
    actual_dates: list
        List of dates for actual prices
    actual_prices: list
        List of actual stock prices
    pred_dates: list
        List of dates for predicted prices
    pred_prices: list
        List of predicted stock prices
    
    Returns:
    --------
    plt.Figure
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual prices
    ax.plot(actual_dates, actual_prices, 'b-', label='Actual Prices')
    
    # Plot predicted prices
    ax.plot(pred_dates, pred_prices, 'r--', label='Predicted Prices')
    
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Stock Price')
    ax.set_title('Tesla Stock Price Prediction')
    
    # Add legend
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def save_plot(fig, filename):
    """Save the plot to a file"""
    fig.savefig(filename)

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics"""
    try:
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def ensure_data_format(data):
    """Validate and format input data"""
    try:
        if not isinstance(data, list):
            raise ValueError("Input data must be a list")
            
        for item in data:
            if not isinstance(item, dict) or 'date' not in item or 'close' not in item:
                raise ValueError("Each data point must have 'date' and 'close' values")
                
        return True
    except Exception as e:
        print(f"Data format error: {str(e)}")
        return False