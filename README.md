Trading Prediction System for NASDAQ 100
Overview
This project develops an advanced trading prediction system for the NASDAQ 100 index (^IXIC), leveraging artificial intelligence to assist traders in making informed investment decisions (Buy, Sell, or Hold). By utilizing a hybrid deep learning model combining Convolutional Neural Networks (CNN), Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU), the system achieves a validation accuracy of 92%. The project spans a 10-year dataset (2015-2025) and includes a user-friendly real-time interface built with Streamlit.
Objectives

Predict trading decisions (Buy, Sell, Hold) for the NASDAQ 100 index with high accuracy.
Provide traders with a real-time, interactive tool to visualize predictions, market trends, and probabilities.
Enhance decision-making by incorporating technical indicators and robust data preprocessing.

Features

Data Collection: Retrieves historical data using yfinance from Yahoo Finance, covering 10 years of NASDAQ 100 data.
Preprocessing: Cleans data, calculates technical indicators (e.g., SMA, EMA, RSI, MACD, Bollinger Bands), and normalizes features using StandardScaler.
Model Architecture: Hybrid CNN-LSTM-GRU model with regularization (Dropout, L2, BatchNormalization) to capture local and temporal patterns.
Prediction Interface: Streamlit-based UI with:
Dynamic ticker selection (default: ^IXIC).
Configurable data range (60 to 180 days).
Real-time predictions with confidence scores.
Visualizations of price trends, technical indicators, and probability distributions.
Adjustable auto-refresh (30 to 300 seconds) and export options.


Performance: Achieves 92% validation accuracy, with detailed evaluation metrics (e.g., confusion matrix, learning curves).

Requirements

Python 3.8+
Libraries: yfinance, pandas, numpy, tensorflow, scikit-learn, streamlit, joblib
Optional: TensorBoard for training visualization

Setup Instructions

Clone the Repository:git clone <repository-url>
cd trading-prediction-nasdaq100


Install Dependencies:pip install -r requirements.txt

Ensure you have a requirements.txt file with the following:yfinance
pandas
numpy
tensorflow
scikit-learn
streamlit
joblib


Download or Prepare Data:
The system uses yfinance to download NASDAQ 100 data (^IXIC). No manual download is required unless you want to use a custom dataset.


Train the Model (Optional):
Run training_module.py to train the CNN-LSTM-GRU model:python training_module.py


Pre-trained weights are provided in the models/ directory.


Launch the Interface:streamlit run interface.py

Access the interface at http://localhost:8501.

Usage

Configure Settings:
In the Streamlit sidebar, select the ticker (^IXIC by default), data range (e.g., 90 days), and refresh interval (e.g., 60 seconds).


Run Predictions:
Click the "Predict" button to generate real-time trading recommendations.
View predictions (Buy/Sell/Hold) with probability distributions.


Visualize Trends:
Explore price trends, technical indicators, and recent data in the dashboard.


Export Results:
Use the "Export Results" button to save predictions and data as a CSV file.


Auto-Refresh:
Enable auto-refresh to update predictions at the specified interval.



Project Structure

data_processing.py: Handles data collection, preprocessing, and feature engineering.
training_module.py: Defines the CNN-LSTM-GRU model, label generation, and training pipeline.
interface.py: Streamlit-based UI for real-time predictions and visualizations.
models/: Directory for pre-trained model weights.
plots/: Directory for generated plots (e.g., confusion matrix, learning curves).
README.md: Project documentation.

Future Improvements

Expand Dataset for "Hold" Class: Address the current scarcity of "Hold" samples to improve model balance.
Real-Time News Sentiment Analysis: Integrate sentiment analysis of financial news to enhance prediction accuracy.
Hyperparameter Optimization: Further tune the model to surpass 95% accuracy, using techniques like Bayesian optimization.
Cloud Deployment: Deploy the model on cloud platforms (e.g., AWS, GCP) for scalability and broader access.
Multi-Index Support: Extend the system to support additional indices (e.g., S&P 500, Dow Jones) for versatility.

Role in the Financial Market
This application plays a transformative role in the financial market by empowering traders with an AI-driven tool for decision-making. Its key contributions include:

Enhanced Decision-Making: Provides actionable insights through accurate predictions and real-time market trend visualizations, enabling traders to act swiftly in volatile markets like the NASDAQ 100.
Competitive Advantage: The use of a hybrid CNN-LSTM-GRU model, combined with advanced preprocessing, gives traders an edge over traditional methods by capturing both short-term and long-term market patterns.
Accessibility for All Traders: The intuitive Streamlit interface makes sophisticated AI tools accessible to both novice and experienced traders, democratizing access to high-quality trading predictions.
Risk Mitigation: By offering probability distributions alongside predictions, the system helps traders assess the confidence of each recommendation, reducing the risk of poor investment decisions.
Scalability and Adaptability: With future cloud deployment, the system can scale to serve a larger user base and adapt to other financial indices, potentially reshaping trading strategies across global markets.

License
This project is licensed under the MIT License. See the LICENSE file for details.

