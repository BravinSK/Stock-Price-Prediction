import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib

from config import MODEL_PATH, SCALER_PATH, SEQUENCE_LENGTH
from utils import create_directory_if_not_exists

def train_lstm_model():
    # Create directory for models
    create_directory_if_not_exists('models')
    
    # Load data
    print("Loading data...")
    data_path = 'data/Tesla.csv'
    df = pd.read_csv(data_path)
    
    # Ensure data is sorted by date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Extract close prices
    close_prices = df['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    # Create sequences for training
    X, y = [], []
    for i in range(len(scaled_data) - SEQUENCE_LENGTH):
        X.append(scaled_data[i:i+SEQUENCE_LENGTH, 0])
        y.append(scaled_data[i+SEQUENCE_LENGTH, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into train and test sets (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss}")
    
    # Save the model and scaler
    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)
    
    print(f"Saving scaler to {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    
    print("Training completed!")
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    create_directory_if_not_exists('static/images')
    plt.savefig('static/images/training_history.png')
    plt.show()

if __name__ == "__main__":
    train_lstm_model()