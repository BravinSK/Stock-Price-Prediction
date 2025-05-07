# Configuration parameters for the Tesla Stock Price Prediction project

# Data parameters
DATA_PATH = 'data/Tesla.csv'
DATE_COLUMN = 'Date'
TARGET_COLUMN = 'Close'
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']

# Model parameters
SEQUENCE_LENGTH = 10  # Input sequence length (10 days of data)
PREDICTION_DAYS = 20  # Maximum number of days to predict ahead
TEST_SIZE = 0.2  # Proportion of data for testing
VALIDATION_SIZE = 0.2  # Proportion of training data for validation

# LSTM model parameters
UNITS = 50  # Number of LSTM units
DROPOUT = 0.2  # Dropout rate
BATCH_SIZE = 32  # Training batch size
EPOCHS = 50  # Training epochs
PATIENCE = 5  # Early stopping patience
LEARNING_RATE = 0.001  # Learning rate

# Saved model path
MODEL_PATH = 'models/pretrained_model.h5'
SCALER_PATH = 'models/scaler.pkl'

# Flask app settings
DEBUG = True
PORT = 5000
HOST = '127.0.0.1'