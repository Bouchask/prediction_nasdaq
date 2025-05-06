import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
import logging
import time
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Function to load the model
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"Failed to load model: {e}")
        return None

# Function to load the scaler
@st.cache_resource
def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        return scaler
    except Exception as e:
        logger.error(f"Error loading scaler: {e}")
        st.warning(f"Failed to load scaler: {e}. Attempting manual scaling on real-time data.")
        return None

# Function to fetch real-time data
def fetch_realtime_data(ticker, period='1d', interval='1d', days=60):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        logger.info(f"Fetched {len(data)} rows for {ticker}")
        return data
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        st.error(f"Failed to fetch data: {e}")
        return None

# Function to preprocess data (compute 13 features and apply scaling)
def preprocess_data(data, scaler, window_size=10):
    try:
        # Reset index to ensure datetime is a column
        data = data.reset_index()
        
        # Log the initial number of rows
        logger.info(f"Initial data rows: {len(data)}")
        
        # Select OHLC and Volume
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Compute technical indicators (same as data_processing.py)
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema_12 - ema_26
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        ma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df['BBM_20'] = ma20
        df['BBU_20'] = ma20 + (2 * std20)
        df['BBL_20'] = ma20 - (2 * std20)
        
        # Drop NaN values
        df = df.dropna()
        
        # Log the number of rows after dropping NaNs
        logger.info(f"Rows after dropping NaNs: {len(df)}")
        
        # Ensure we have enough data for 10 timesteps
        if len(df) < window_size:
            raise ValueError(
                f"Not enough data after preprocessing: {len(df)} rows. "
                f"The model requires {window_size} rows, but only {len(df)} are available after computing indicators. "
                "This may be due to insufficient trading days in the fetched period or holidays/weekends. "
                "Try increasing the 'days' parameter (e.g., fetch 90 days) to ensure at least 30 trading days are available."
            )
        
        # Select the latest 10 timesteps
        df = df.tail(window_size)
        
        # Define features (same as data_processing.py)
        features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_10', 'EMA_10', 'RSI',
            'MACD_12_26_9', 'MACDs_12_26_9',
            'BBL_20', 'BBM_20', 'BBU_20'
        ]
        if len(features) != 13:
            raise ValueError(f"Expected 13 features, got {len(features)}")
        
        # Convert to numpy array
        X = df[features].values
        X = X.astype(np.float32)
        
        # Reshape to (1, 10, 13)
        X = X.reshape(1, window_size, len(features))
        logger.info(f"Data shape before scaling: {X.shape}")
        
        # Apply scaling
        # The scaler expects 130 features (10 timesteps * 13 features)
        X_reshaped = X.reshape(X.shape[0], -1)  # Shape: (1, 10 * 13) = (1, 130)
        logger.info(f"Data shape after reshaping for scaling: {X_reshaped.shape}")
        
        if scaler:
            X_scaled = scaler.transform(X_reshaped)
        else:
            # Fallback: Manual scaling on real-time data
            st.warning("Performing manual scaling on real-time data (results may differ from training).")
            manual_scaler = StandardScaler()
            X_scaled = manual_scaler.fit_transform(X_reshaped)
        
        # Reshape back to (1, 10, 13)
        X = X_scaled.reshape(X.shape)
        logger.info(f"Data shape after scaling: {X.shape}")
        
        return X, df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        st.error(f"Failed to preprocess data: {e}")
        return None, None

# Function to make prediction
def make_prediction(model, X):
    try:
        prediction = model.predict(X, verbose=0)
        probs = prediction[0]
        predicted_class = np.argmax(probs)
        class_names = ['Sell', 'Hold', 'Buy']
        return class_names[predicted_class], probs
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        st.error(f"Failed to make prediction: {e}")
        return None, None

# Streamlit app
def main():
    st.set_page_config(page_title="Trading Model Interface", layout="wide")
    st.title("Real-Time Trading Prediction Interface")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Stock Ticker (e.g., ^IXIC)", value="^IXIC")
        model_path = st.text_input("Model Path", value="models/model_final_20250501_015240.h5")
        scaler_path = st.text_input("Scaler Path", value="models/scaler.pkl")
        learning_curves_path = st.text_input("Learning Curves Image Path", value="plots/learning_curves_20250501_015240.png")
        days_to_fetch = st.slider("Days of Data to Fetch", 30, 120, 60)
        refresh_interval = st.slider("Refresh Interval (seconds)", 30, 300, 60)
        predict_button = st.button("Predict")
    
    # Load model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)
    
    # Display learning curves image
    if os.path.exists(learning_curves_path):
        st.subheader("Training History (Accuracy and Loss)")
        st.image(learning_curves_path, caption="Learning Curves", use_container_width=True)
    else:
        st.error(f"Learning curves image not found at {learning_curves_path}")
    
    # Real-time prediction
    if predict_button and model:
        with st.spinner("Fetching and processing data..."):
            # Fetch real-time data
            data = fetch_realtime_data(ticker, days=days_to_fetch)
            if data is not None:
                # Preprocess data
                X, df_processed = preprocess_data(data, scaler)
                if X is not None:
                    # Make prediction
                    predicted_class, probs = make_prediction(model, X)
                    if predicted_class is not None:
                        st.subheader("Prediction Results")
                        st.write(f"**Predicted Action**: {predicted_class}")
                        st.write("**Probabilities**:")
                        st.write(f"- Sell: {probs[0]:.4f}")
                        st.write(f"- Hold: {probs[1]:.4f}")
                        st.write(f"- Buy: {probs[2]:.4f}")
                        
                        # Display recent data
                        st.subheader("Recent Data (Last 10 Days)")
                        st.dataframe(df_processed)
    
    # Auto-refresh for real-time updates
    if st.checkbox("Enable Auto-Refresh", value=False):
        st.write(f"Refreshing every {refresh_interval} seconds...")
        st.experimental_rerun()
        time.sleep(refresh_interval)

if __name__ == "__main__":
    main()