# ✅ Importation des librairies
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import sys
import os
import joblib
import logging

# ✅ Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Paramètres
symbol = '^IXIC'  # Symbole pour NASDAQ
period = 10  # Période en années
start_date = (datetime.today() - timedelta(days=period*365)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')
output_dir = "data"  # Dossier pour sauvegarde CSV
models_dir = "models"  # Dossier pour sauvegarde du scaler
os.makedirs(output_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# ✅ Télécharger les données avec yfinance
def download_data(symbol, start_date, end_date, retries=3, delay=10):
    """
    Télécharge les données historiques depuis Yahoo Finance avec gestion des erreurs.
    """
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval='1d', auto_adjust=False)
            if df.empty:
                raise ValueError("No data downloaded. DataFrame is empty.")
            logger.info(f"Données téléchargées pour {symbol}: {len(df)} lignes")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement: {e}")
            if attempt < retries - 1:
                logger.info(f"Nouvelle tentative dans {delay} secondes...")
                time.sleep(delay)
            else:
                logger.error(f"Toutes les tentatives ont échoué pour {symbol}.")
                return None

# ✅ Sauvegarder les données en CSV
def save_data(df, symbol, suffix=""):
    """
    Sauvegarde le DataFrame dans un fichier CSV.
    """
    filename = os.path.join(output_dir, f"{symbol}_{suffix}data.csv")
    df.to_csv(filename)
    logger.info(f"Données sauvegardées dans {filename}")

# ✅ Sauvegarder les séquences normalisées (X et y)
def save_scaled_data(X, y, features, dates, symbol, scaler):
    """
    Sauvegarde les séquences normalisées X et y dans un fichier CSV.
    Inclut l'index temporel et sauvegarde le scaler.
    """
    # Aplatir X pour créer un DataFrame
    X_reshaped = X.reshape(X.shape[0], -1)
    # Créer des noms de colonnes : feature_jour
    columns = [f"{feat}_t{i}" for i in range(X.shape[1]) for feat in features]
    X_df = pd.DataFrame(X_reshaped, columns=columns)
    # Ajouter la colonne cible y et les dates
    X_df['Target'] = y
    X_df['Date'] = dates[-X.shape[0]:]  # Correspond aux séquences générées
    # Sauvegarder
    save_data(X_df, symbol, "scaled_")
    # Sauvegarder le scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    try:
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler sauvegardé dans {scaler_path}")
        # Test loading the scaler to verify
        test_scaler = joblib.load(scaler_path)
        logger.info("Scaler chargé avec succès après sauvegarde (vérification OK)")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde ou du chargement du scaler: {e}")
        raise

# ✅ Tenter le téléchargement des données
df = download_data(symbol, start_date, end_date)
if df is None or df.empty:
    logger.warning(f"Échec avec la plage de {period} ans. Tentative avec 1 an de données...")
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = download_data(symbol, start_date, end_date)
    if df is None or df.empty:
        logger.error(f"Impossible de télécharger les données pour {symbol}, même avec 1 an.")
        sys.exit(1)

# Sauvegarder les données brutes
save_data(df, symbol, "raw_")

# ✅ Nettoyage initial
df.dropna(inplace=True)
if df.empty:
    logger.error("DataFrame vide après suppression des NaN.")
    sys.exit(1)

# ✅ Calcul des indicateurs techniques
# Moyenne mobile simple (SMA) et exponentielle (EMA)
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

# RSI (Relative Strength Index)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD_12_26_9'] = ema_12 - ema_26
df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()

# Bandes de Bollinger
ma20 = df['Close'].rolling(window=20).mean()
std20 = df['Close'].rolling(window=20).std()
df['BBM_20'] = ma20
df['BBU_20'] = ma20 + (2 * std20)
df['BBL_20'] = ma20 - (2 * std20)

# Colonne cible (prix de clôture du jour suivant)
df['Target'] = df['Close'].shift(-1)

# Supprimer les lignes avec NaN après calcul des indicateurs
df.dropna(inplace=True)
if df.empty:
    logger.error("DataFrame vide après calcul des indicateurs.")
    sys.exit(1)

# Sauvegarder les données avec les indicateurs
save_data(df, symbol, "with_indicators_")

# ✅ Liste des features
features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'SMA_10', 'EMA_10', 'RSI',
    'MACD_12_26_9', 'MACDs_12_26_9',
    'BBL_20', 'BBM_20', 'BBU_20'
]

# ✅ Construction des séquences (10 jours d'inputs pour prédire 1 jour)
window_size = 10
X, y = [], []
for i in range(window_size, len(df)):
    X.append(df[features].iloc[i-window_size:i].values)
    y.append(df['Target'].iloc[i])

X = np.array(X)
y = np.array(y)
dates = df.index  # Sauvegarder l'index temporel

# Vérifier si X et y sont vides
if X.size == 0 or y.size == 0:
    logger.error("Aucune séquence générée. Vérifiez les données ou réduisez la taille de la fenêtre.")
    sys.exit(1)

# ✅ Normalisation
X_reshaped = X.reshape(X.shape[0], -1)  # Aplatir pour le scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)
X = X_scaled.reshape(X.shape)  # Remettre en forme originale

# ✅ Sauvegarder les données normalisées
save_scaled_data(X, y, features, dates, symbol, scaler)

# ✅ Split Train/Validation/Test (70/15/15)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.176, shuffle=False)  # 0.176*0.85 ~= 0.15

# ✅ Sauvegarde des ensembles pour le script d'entraînement
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
np.save(os.path.join(output_dir, 'dates.npy'), dates[-X.shape[0]:])  # Dates correspondant aux séquences

# ✅ Affichage des résultats
logger.info(f"\nShape X: {X.shape}")
logger.info(f"Shape y: {y.shape}")
logger.info("\nRépartition des données:")
logger.info(f"Train: {X_train.shape}")
logger.info(f"Validation: {X_val.shape}")
logger.info(f"Test: {X_test.shape}")
logger.info("\nExemple X[0] (séquence):")
logger.info(pd.DataFrame(X[0], columns=features).to_string())
logger.info(f"\ny[0] (target): {y[0]}")