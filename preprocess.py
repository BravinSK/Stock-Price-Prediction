import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from config import (
    DATA_PATH, DATE_COLUMN, TARGET_COLUMN, FEATURE_COLUMNS,
    SEQUENCE_LENGTH, SCALER_PATH, TEST_SIZE
)
from utils import create_directory_if_not_exists

def load_data(file_path=DATA_PATH):
    """Load and validate the dataset"""
    try:
        df = pd.read_csv(file_path)
        required_columns = ['Date', 'Close']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Dataset missing required columns: Date, Close")
            
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def preprocess_data(df):
    """Preprocess the data for training"""
    try:
        # Sort by date
        df = df.sort_values('Date')
        
        # Extract close prices
        close_prices = df['Close'].values.reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        
        return scaled_data, scaler
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")

def prepare_data_for_training(scaled_data, sequence_length=SEQUENCE_LENGTH):
    """Prepare sequences for LSTM training"""
    try:
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length), 0])
            y.append(scaled_data[i + sequence_length, 0])
            
        return np.array(X), np.array(y)
    except Exception as e:
        raise Exception(f"Error preparing sequences: {str(e)}")

def scale_features(df, train_only=False, scaler=None):
    """
    Scale the features using MinMaxScaler
    
    Parameters:
    -----------
    df: pd.DataFrame
        DataFrame with features to scale
    train_only: bool
        If True, use only training data for fitting the scaler
    scaler: MinMaxScaler, optional
        Pre-fitted scaler (for transformation only)
    
    Returns:
    --------
    tuple
        (scaled_data, scaler, columns)
    """
    # Extract features
    features = df[FEATURE_COLUMNS].values
    
    if scaler is None:
        # Create and fit the scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        if train_only:
            # Use only first 80% (or as per TEST_SIZE) for fitting
            n_train = int(len(features) * (1 - TEST_SIZE))
            scaler.fit(features[:n_train])
        else:
            scaler.fit(features)
            
        # Save the scaler
        create_directory_if_not_exists(os.path.dirname(SCALER_PATH))
        joblib.dump(scaler, SCALER_PATH)
    
    # Transform data
    scaled_data = scaler.transform(features)
    
    return scaled_data, scaler, FEATURE_COLUMNS

def create_sequences(data, target_idx=3):
    """
    Create input sequences and targets for LSTM model
    
    Parameters:
    -----------
    data: np.ndarray
        Scaled feature data
    target_idx: int
        Index of the target column in data
    
    Returns:
    --------
    tuple
        (X, y) where X is input sequences and y is target values
    """
    X = []
    y = []
    
    for i in range(SEQUENCE_LENGTH, len(data)):
        X.append(data[i-SEQUENCE_LENGTH:i])
        y.append(data[i, target_idx])
    
    return np.array(X), np.array(y)

def prepare_data_for_training(df=None):
    """
    Prepare the data for training
    
    Parameters:
    -----------
    df: pd.DataFrame, optional
        DataFrame containing the stock data
    
    Returns:
    --------
    tuple
        (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    if df is None:
        df = load_data()
        df = preprocess_data(df)
    
    if df is None or df.empty:
        print("Error: No data available for training")
        return None
    
    # Get the scaled data
    scaled_data, scaler, _ = scale_features(df, train_only=True)
    
    # Create sequences
    X, y = create_sequences(scaled_data)
    
    # Split data into train, validation, and test sets
    n_train = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Split training data into train and validation
    n_val = int(len(X_train) * 0.2)
    X_train, X_val = X_train[:-n_val], X_train[-n_val:]
    y_train, y_val = y_train[:-n_val], y_train[-n_val:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

def prepare_input_for_prediction(input_data):
    """
    Prepare input data for prediction
    
    Parameters:
    -----------
    input_data: list
        List of dictionaries containing date and close price
    
    Returns:
    --------
    tuple
        (scaled_data, dates)
    """
    # Load the scaler
    try:
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"Error: Scaler not found at {SCALER_PATH}")
        return None
    
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Extract dates
    dates = df['date'].tolist() if 'date' in df.columns else None
    
    # Create feature columns if not present
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            if col == 'Close':
                df[col] = df['close'] if 'close' in df.columns else 0
            else:
                df[col] = df['close'] if 'close' in df.columns else 0
    
    # Scale the data
    scaled_data = scaler.transform(df[FEATURE_COLUMNS].values)
    
    return scaled_data, dates

def prepare_sequence_for_prediction(scaled_data):
    """
    Prepare input sequence for prediction
    
    Parameters:
    -----------
    scaled_data: np.ndarray
        Scaled feature data
    
    Returns:
    --------
    np.ndarray
        Input sequence for LSTM model
    """
    if len(scaled_data) < SEQUENCE_LENGTH:
        print(f"Error: Input data must have at least {SEQUENCE_LENGTH} rows")
        return None
    
    # Use the last SEQUENCE_LENGTH data points as input
    sequence = scaled_data[-SEQUENCE_LENGTH:]
    
    # Reshape for LSTM input [samples, time steps, features]
    return np.reshape(sequence, (1, sequence.shape[0], sequence.shape[1]))