import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
import kagglehub

class EEGConfig:
    def __init__(self):
        # Paramètres du dataset
        self.n_channels = 22
        self.n_classes = 4
        self.target_length = 201  # Longueur fixe après groupement par epoch
        
        # Paramètres de fenêtrage glissant
        self.window_size = 50
        self.stride = 10
        
        # Paramètres du modèle
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 1e-3
        self.dropout_rate = 0.3
        self.l2_reg = 0.002
        
        # Colonnes EEG
        self.eeg_columns = [
            'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 
            'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 
            'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'
        ]

def load_and_preprocess_data_improved(csv_path, config):
    """
    Préprocessing amélioré inspiré du modèle performant
    """
    print("Chargement des données...")
    df = pd.read_csv(csv_path)
    
    # Conversion en numérique et suppression des NaN
    df[config.eeg_columns] = df[config.eeg_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()
    
    print(f"Dataset shape après nettoyage: {df.shape}")
    print("Distribution des classes avant fenêtrage:")
    print(df['label'].value_counts())
    
    # **CHANGEMENT CLÉ 1: Groupement par epoch uniquement**
    grouped = df.groupby('epoch')
    X, y = [], []
    
    for epoch, group in grouped:
        # Transposer pour avoir (channels, time_points)
        data = group[config.eeg_columns].values.T
        
        # Padding/truncation à la longueur cible
        if data.shape[1] > config.target_length:
            data = data[:, :config.target_length]
        elif data.shape[1] < config.target_length:
            pad_width = config.target_length - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')
        
        X.append(data)
        y.append(group['label'].iloc[0])
    
    X = np.array(X)  # Shape: (n_samples, n_channels, n_time_points)
    y = np.array(y)
    
    print(f"X shape après groupement par epoch: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y

def apply_sliding_window_with_frequency(X, y, config):
    """
    **CHANGEMENT CLÉ 2: Fenêtrage glissant + features fréquentielles**
    """
    print("Application du fenêtrage glissant avec features fréquentielles...")
    
    n_samples, n_channels, n_time_points = X.shape
    
    # **Normalisation par échantillon et par canal**
    X_normalized = np.zeros_like(X)
    for i in range(n_samples):
        for ch in range(n_channels):
            scaler = StandardScaler()
            X_normalized[i, ch, :] = scaler.fit_transform(X[i, ch, :].reshape(-1, 1)).flatten()
    
    # **Features fréquentielles via FFT**
    X_freq = np.zeros((n_samples, n_channels, n_time_points // 2 + 1))
    for i in range(n_samples):
        for ch in range(n_channels):
            fft_result = fft(X_normalized[i, ch, :])
            X_freq[i, ch, :] = np.abs(fft_result[:n_time_points // 2 + 1])
    
    # **Fenêtrage glissant**
    X_windows, y_windows = [], []
    
    for i in range(n_samples):
        data = X_normalized[i]
        freq_data = X_freq[i]
        
        num_windows = (n_time_points - config.window_size) // config.stride + 1
        
        for w in range(num_windows):
            start = w * config.stride
            end = start + config.window_size
            
            # Fenêtre temporelle
            window = data[:, start:end]
            
            # Fenêtre fréquentielle correspondante
            freq_start = (start * (n_time_points // 2 + 1)) // n_time_points
            freq_end = (end * (n_time_points // 2 + 1)) // n_time_points
            freq_window = freq_data[:, freq_start:freq_end]
            
            # Ajustement de la taille de la fenêtre fréquentielle
            if freq_window.shape[1] < config.window_size:
                freq_window = np.pad(freq_window, ((0, 0), (0, config.window_size - freq_window.shape[1])), mode='constant')
            elif freq_window.shape[1] > config.window_size:
                freq_window = freq_window[:, :config.window_size]
            
            # **Combinaison des domaines temporel et fréquentiel**
            combined = np.stack([window, freq_window], axis=-1)
            X_windows.append(combined)
            y_windows.append(y[i])
    
    X_windows = np.array(X_windows)
    y_windows = np.array(y_windows)
    
    print(f"X_windows shape avant reshape: {X_windows.shape}")
    print(f"Nombre d'échantillons après fenêtrage: {len(y_windows)}")
    
    # Reshape pour le modèle: (samples, time_steps, channels * features)
    X_windows = X_windows.transpose(0, 2, 1, 3)
    X_windows = X_windows.reshape(X_windows.shape[0], X_windows.shape[1], X_windows.shape[2] * X_windows.shape[3])
    
    print(f"X_windows shape final: {X_windows.shape}")
    
    return X_windows, y_windows

def create_improved_cnn_lstm_model(input_shape, num_classes, config):
    """
    **CHANGEMENT CLÉ 3: Architecture simplifiée mais efficace**
    """
    input1 = layers.Input(shape=input_shape)
    
    # Première couche Conv1D
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='elu', 
                      kernel_regularizer=l2(config.l2_reg))(input1)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout_rate)(x)
    
    # Deuxième couche Conv1D
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='elu', 
                      kernel_regularizer=l2(config.l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    
    # Couches LSTM
    x = layers.LSTM(128, return_sequences=True, dropout=config.dropout_rate)(x)
    x = layers.LSTM(64, dropout=config.dropout_rate)(x)
    
    # Couche dense finale
    x = layers.Dense(64, activation='elu', kernel_regularizer=l2(config.l2_reg))(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    return Model(inputs=input1, outputs=x)

def main_improved(csv_path):
    """
    Pipeline amélioré inspiré du modèle performant
    """
    config = EEGConfig()
    
    # Chargement et preprocessing initial
    X, y = load_and_preprocess_data_improved(csv_path, config)
    
    # Application du fenêtrage glissant avec features fréquentielles
    X_windows, y_windows = apply_sliding_window_with_frequency(X, y, config)
    
    # Encodage des labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_windows)
    y_categorical = tf.keras.utils.to_categorical(y_encoded)
    
    print(f"Shape des labels catégorielles: {y_categorical.shape}")
    
    # Division train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_windows, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    
    print(f"Train: {X_train.shape}, {y_train.shape}")
    print(f"Test: {X_test.shape}, {y_test.shape}")
    
    # Création et compilation du modèle
    model = create_improved_cnn_lstm_model(X_train.shape[1:], config.n_classes, config)
    
    optimizer = Adam(learning_rate=config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nArchitecture du modèle amélioré:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.7, patience=8, min_lr=1e-6, verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_improved_eeg_model.h5', monitor='val_accuracy', 
            save_best_only=True, verbose=1
        )
    ]
    
    # Entraînement
    print("\nDébut de l'entraînement du modèle amélioré...")
    history = model.fit(
        X_train, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history, label_encoder

# Utilisation
if __name__ == "__main__":
    csv_dir_path = kagglehub.dataset_download("aymanmostafa11/eeg-motor-imagery-bciciv-2a")
    csv_path = csv_dir_path + '/BCICIV_2a_all_patients.csv'
    model, history, label_encoder = main_improved(csv_path)