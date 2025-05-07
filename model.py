import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from config import (
    SEQUENCE_LENGTH, UNITS, DROPOUT, BATCH_SIZE, EPOCHS,
    PATIENCE, MODEL_PATH, LEARNING_RATE
)
from preprocess import load_data, preprocess_data, prepare_data_for_training
from utils import create_directory_if_not_exists, calculate_metrics

def create_lstm_model(input_shape):
    """
    Create LSTM model for stock price prediction
    
    Parameters:
    -----------
    input_shape: tuple
        Shape of input data (time steps, features)
    
    Returns:
    --------
    tensorflow.keras.Model
        Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(units=UNITS, 
                   return_sequences=True, 
                   input_shape=input_shape))
    model.add(Dropout(DROPOUT))
    
    # Second LSTM layer
    model.add(LSTM(units=UNITS, 
                   return_sequences=False))
    model.add(Dropout(DROPOUT))
    
    # Output layer
    model.add(Dense(units=1))
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    model.summary()
    
    return model

def ensure_model_exists(data_path='data/Tesla.csv'):
    """Ensure a trained model exists, create if not"""
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'pretrained_model.h5')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("No pre-trained model found. Training new model...")
        if os.path.exists(data_path):
            train_model(data_path=data_path,
                        model_save_path=model_path,
                        scaler_save_path=scaler_path)
            return True
        else:
            print(f"Error: Training data not found at {data_path}")
            return False
    return True

def train_model(data_path='data/Tesla.csv', model_save_path='models/pretrained_model.h5', scaler_save_path='models/scaler.pkl'):
    """Train the LSTM model"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        # Load and prepare data using preprocessing functions
        df = load_data(data_path)
        scaled_data, scaler = preprocess_data(df)
        X, y = prepare_data_for_training(scaled_data)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Build and train the model
        model = create_lstm_model((SEQUENCE_LENGTH, 1))
        history = model.fit(X, y, 
                          epochs=EPOCHS, 
                          batch_size=BATCH_SIZE, 
                          validation_split=0.1, 
                          verbose=1)
        
        # Save the model and scaler
        model.save(model_save_path)
        joblib.dump(scaler, scaler_save_path)
        
        # Calculate metrics
        metrics = {
            'RMSE': np.sqrt(model.evaluate(X, y, verbose=0)),
            'final_loss': history.history['loss'][-1]
        }
        
        return model, history, metrics
        
    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        return None, None, None

def plot_training_history(history):
    """
    Plot training and validation loss
    
    Parameters:
    -----------
    history: tensorflow.keras.callbacks.History
        Training history object
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    create_directory_if_not_exists('static/images')
    plt.savefig('static/images/training_history.png')
    plt.close()

if __name__ == "__main__":
    if ensure_model_exists():
        print("Model is ready for use.")
    else:
        print("Model initialization failed.")