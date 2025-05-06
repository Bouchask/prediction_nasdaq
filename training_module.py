import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import os

# ✅ Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ✅ Vérifier la disponibilité du GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    logger.info(f"GPU détecté : {physical_devices}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    logger.warning("Aucun GPU détecté, utilisation du CPU.")

# ✅ Validation des données d'entrée
def validate_input_data(X, y):
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("X et y doivent être des tableaux NumPy.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Nombre d'échantillons mismatch: X ({X.shape[0]}) vs y ({y.shape[0]}).")
    if np.any(np.isnan(X)) or np.any(np.isnan(y)):
        raise ValueError("Les données contiennent des NaN.")
    if X.ndim != 3:
        raise ValueError(f"X doit être 3D (samples, timesteps, features), trouvé: {X.shape}")
    if not np.all(np.isin(y, [0, 1, 2])):
        raise ValueError(f"Labels must be in [0, 1, 2], found: {np.unique(y)}")
    logger.info(f"Validation des données OK: X shape {X.shape}, y shape {y.shape}")

# ✅ Split data into train (70%), validation (15%), and test (15%)
def split_data(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, data_dir="data"):
    """
    Splits data into train, validation, and test sets with stratification.
    Saves splits to data_dir.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    # First split: train + validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=42
    )
    
    # Second split: train vs validation
    val_size = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
    )
    
    # Save splits
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(data_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(data_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
    
    logger.info(f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    logger.info(f"Train label distribution: {np.bincount(y_train)}")
    logger.info(f"Val label distribution: {np.bincount(y_val)}")
    logger.info(f"Test label distribution: {np.bincount(y_test)}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ✅ Conversion des données en tf.data.Dataset
def create_dataset(X, y, batch_size=32, shuffle=True):
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    logger.info(f"Label values in dataset: {np.unique(y)}")
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=X.shape[0])
    dataset = dataset.batch(batch_size)
    return dataset

# ✅ Création des étiquettes Buy/Sell/Hold
def create_trading_labels(y, close_prices, threshold=0.001):
    """
    Génère des étiquettes Buy (2), Sell (0), Hold (1) basées sur la variation du prix.
    Explicitly ensures Hold labels for small variations.
    """
    variations = (y - close_prices) / close_prices
    logger.info(f"Variations range: min={np.min(variations)}, max={np.max(variations)}, mean={np.mean(variations)}")
    labels = np.ones(len(y), dtype=np.int32)  # Par défaut: Hold (1)
    labels[variations > threshold] = 2       # Buy (2)
    labels[variations < -threshold] = 0      # Sell (0)
    labels[np.abs(variations) <= threshold] = 1  # Explicitly set Hold for small variations
    logger.info(f"Label distribution: {np.bincount(labels)}")
    return labels

# ✅ Création du modèle avec régularisation renforcée

def build_model(input_shape, conv_filters=32, lstm_units=32, gru_units=16, dense_units=16, dropout_rate=0.5):
    model = Sequential([
        Conv1D(filters=conv_filters, kernel_size=2, activation='relu', input_shape=input_shape, padding='same',
               kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(dropout_rate),  # ✅ Dropout après BN
        
        LSTM(lstm_units, return_sequences=True, recurrent_dropout=0.3, unroll=True),
        BatchNormalization(),
        Dropout(dropout_rate),  # ✅ Dropout après BN
        
        GRU(gru_units, recurrent_dropout=0.3, unroll=True),
        BatchNormalization(),
        Dropout(dropout_rate),  # ✅ Dropout après BN
        
        Dense(dense_units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02)),
        BatchNormalization(),
        Dropout(dropout_rate),  # ✅ Dropout après BN
        
        Dense(3, activation='softmax', dtype='float32')  # 3 classes: Sell, Hold, Buy
    ])
    return model


# ✅ Sauvegarde des métriques et matrice de confusion
def save_metrics(y_true, y_pred, filename, class_names=['Sell', 'Hold', 'Buy']):
    labels = [0, 1, 2]  # Sell, Hold, Buy
    report = classification_report(y_true, y_pred, target_names=class_names, labels=labels, output_dict=False, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    with open(filename, 'w') as f:
        f.write("Rapport de classification:\n")
        f.write(report)
        f.write("\nMatrice de confusion:\n")
        f.write(str(cm))
    logger.info(f"Métriques sauvegardées dans {filename}")
    return cm

# ✅ Plot de la matrice de confusion
def plot_confusion_matrix(cm, timestamp, class_names=['Sell', 'Hold', 'Buy']):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plot_path = f"plots/confusion_matrix_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée dans {plot_path}")

# ✅ Plot des courbes d'apprentissage
def plot_learning_curves(history, timestamp):
    plt.figure(figsize=(12, 5))
    
    # Perte
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    # Précision
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = f"plots/learning_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Courbes d'apprentissage sauvegardées dans {plot_path}")

# ✅ Paramètres configurables
batch_size = 32
epochs = 100
conv_filters = 32  # Réduit pour éviter le surapprentissage
lstm_units = 32   # Réduit
gru_units = 16    # Réduit
dense_units = 16  # Réduit
dropout_rate = 0.5  # Augmenté pour plus de régularisation
threshold = 0.001   # Réduit pour inclure plus de Hold

try:
    # ✅ Charger les données pré-divisées
    data_dir = "data"
    X_train_old = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train_old = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val_old = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val_old = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test_old = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test_old = np.load(os.path.join(data_dir, 'y_test.npy'))

    # ✅ Concaténer les données pour re-diviser
    X = np.concatenate([X_train_old, X_val_old, X_test_old], axis=0)
    y = np.concatenate([y_train_old, y_val_old, y_test_old], axis=0)
    logger.info(f"Concatenated data: X shape={X.shape}, y shape={y.shape}")

    # ✅ Créer les étiquettes Buy/Sell/Hold pour stratification
    close_prices = X[:, -1, 3]  # Dernière timestep, feature 'Close'
    y_labels = create_trading_labels(y, close_prices, threshold)

    # ✅ Split data into train (70%), val (15%), test (15%)
    X_train, y_train_labels, X_val, y_val_labels, X_test, y_test_labels = split_data(
        X, y_labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, data_dir=data_dir
    )

    # ✅ Validation des données
    validate_input_data(X_train, y_train_labels)
    validate_input_data(X_val, y_val_labels)
    validate_input_data(X_test, y_test_labels)

    # ✅ Calcul des poids des classes
    unique_classes = np.unique(y_train_labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train_labels)
    class_weight_dict = {i: class_weights[np.where(unique_classes == i)[0][0]] for i in unique_classes}
    for i in range(3):
        if i not in class_weight_dict:
            class_weight_dict[i] = 1.0
    logger.info(f"Poids des classes: {class_weight_dict}")

    # ✅ Création des datasets
    train_dataset = create_dataset(X_train, y_train_labels, batch_size, shuffle=True)
    val_dataset = create_dataset(X_val, y_val_labels, batch_size, shuffle=False)
    test_dataset = create_dataset(X_test, y_test_labels, batch_size, shuffle=False)

    # ✅ Création du modèle
    model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                        conv_filters=conv_filters,
                        lstm_units=lstm_units,
                        gru_units=gru_units,
                        dense_units=dense_units,
                        dropout_rate=dropout_rate)

    # ✅ Compilation avec mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # ✅ Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)  # Patience augmentée
    checkpoint = ModelCheckpoint(f"models/model_{timestamp}.h5", monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir=f"logs/fit/{timestamp}", histogram_freq=1)

    # ✅ Entraînement
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stop, checkpoint, tensorboard],
        class_weight=class_weight_dict,
        verbose=1
    )
    logger.info("Entraînement terminé.")

    # ✅ Sauvegarde du modèle final
    model.save(f"models/model_final_{timestamp}.h5")
    logger.info(f"Modèle sauvegardé sous models/model_final_{timestamp}.h5")

    # ✅ Évaluation sur l'ensemble de test
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    cm = save_metrics(y_test_labels, y_pred, f"models/performance_{timestamp}.txt")
    plot_confusion_matrix(cm, timestamp)
    plot_learning_curves(history, timestamp)

except Exception as e:
    logger.error(f"Erreur: {e}")
    raise

finally:
    tf.keras.mixed_precision.set_global_policy('float32')