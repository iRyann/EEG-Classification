import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub


# Configuration du modèle adaptée aux données CSV
class EEGConfig:
    def __init__(self):
        # Paramètres du dataset basés sur vos données
        self.n_channels = 22  # Canaux EEG (sans patient, time, label, epoch)
        self.n_classes = 4    # 4 classes: tongue, foot, right, left
        self.time_range = (-0.1, 2)  # Plage temporelle
        self.trial_duration = 1.9  # 0.7 - (-0.1) = 0.8 secondes
        
        # Paramètres du modèle
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 1e-3
        self.dropout_rate = 0.5
        
        # Colonnes EEG dans l'ordre du CSV
        self.eeg_columns = [
            'EEG-Fz', 'EEG-0', 'EEG-1', 'EEG-2', 'EEG-3', 'EEG-4', 'EEG-5', 
            'EEG-C3', 'EEG-6', 'EEG-Cz', 'EEG-7', 'EEG-C4', 'EEG-8', 'EEG-9', 
            'EEG-10', 'EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-Pz', 'EEG-15', 'EEG-16'
        ]

def load_and_preprocess_data(csv_path, config):
    """
    Charge et préprocesse les données CSV
    """
    print("Chargement des données...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Colonnes: {df.columns.tolist()}")
    print(f"Patients uniques: {df['patient'].nunique()}")
    print(f"Labels uniques: {df['label'].unique()}")
    
    # Encodage des labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    
    print(f"Mapping des labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Groupement par patient et epoch pour créer les séquences temporelles
    print("Création des séquences temporelles...")
    
    sequences = []
    labels = []
    patients = []
    
    # Grouper par patient et epoch
    for (patient, epoch), group in df.groupby(['patient', 'epoch']):
        # Trier par temps pour assurer l'ordre temporel
        group_sorted = group.sort_values('time')
        
        # Extraire les données EEG
        eeg_data = group_sorted[config.eeg_columns].values
        
        # Vérifier que nous avons une séquence complète
        if len(eeg_data) > 10:  # Seuil minimum pour une séquence valide
            sequences.append(eeg_data)
            labels.append(group_sorted['label_encoded'].iloc[0])
            patients.append(patient)
    
    print(f"Nombre de séquences créées: {len(sequences)}")
    
    # Padding/truncation des séquences pour avoir une longueur uniforme
    max_length = max(len(seq) for seq in sequences)
    min_length = min(len(seq) for seq in sequences)
    target_length = int(np.percentile([len(seq) for seq in sequences], 75))  # 75e percentile
    
    print(f"Longueurs des séquences - Min: {min_length}, Max: {max_length}, Target: {target_length}")
    
    X = np.zeros((len(sequences), target_length, config.n_channels))
    
    for i, seq in enumerate(sequences):
        if len(seq) >= target_length:
            # Truncation
            X[i] = seq[:target_length]
        else:
            # Padding avec la dernière valeur
            X[i, :len(seq)] = seq
            X[i, len(seq):] = seq[-1]
    
    y = np.array(labels)
    patients_array = np.array(patients)
    
    print(f"Shape finale: X={X.shape}, y={y.shape}")
    
    return X, y, patients_array, label_encoder, target_length

def create_cnn_lstm_model(config, sequence_length):
    """
    Crée un modèle hybride CNN-LSTM adapté aux données EEG temporelles
    """
    
    # Input layer
    input_layer = layers.Input(shape=(sequence_length, config.n_channels), name='eeg_input')
    
    # === PARTIE CNN POUR EXTRACTION DE FEATURES SPATIALES ===
    # Première couche de convolution temporelle
    x = layers.Conv1D(filters=32, kernel_size=7, strides=1, padding='same', 
                      activation='relu', name='conv1d_1')(input_layer)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool_1')(x)
    x = layers.SpatialDropout1D(0.3, name='spatial_dropout_1')(x)
    
    # Deuxième couche de convolution
    x = layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='same', 
                      activation='relu', name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool_2')(x)
    x = layers.SpatialDropout1D(0.3, name='spatial_dropout_2')(x)
    
    # Troisième couche de convolution
    x = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', 
                      activation='relu', name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='pool_3')(x)
    x = layers.SpatialDropout1D(0.3, name='spatial_dropout_3')(x)
    
    # === PARTIE LSTM POUR DÉPENDANCES TEMPORELLES ===
    # Première couche LSTM bidirectionnelle
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        name='bilstm_1'
    )(x)
    
    # Deuxième couche LSTM
    x = layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
                    name='lstm_2')(x)
    
    # === ATTENTION MECHANISM (OPTIONNEL) ===
    # Couche d'attention pour se concentrer sur les parties importantes
    # x = layers.Dense(64, activation='tanh', name='attention_dense')(x)
    # x = layers.Dense(1, activation='softmax', name='attention_weights')(x)
    
    # === COUCHES DE CLASSIFICATION ===
    # Couches denses avec régularisation
    x = layers.Dense(128, activation='relu', name='dense_1')(x)
    x = layers.BatchNormalization(name='bn_dense_1')(x)
    x = layers.Dropout(config.dropout_rate, name='dropout_dense_1')(x)
    
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.BatchNormalization(name='bn_dense_2')(x)
    x = layers.Dropout(config.dropout_rate, name='dropout_dense_2')(x)
    
    x = layers.Dense(32, activation='relu', name='dense_3')(x)
    x = layers.Dropout(0.3, name='dropout_dense_3')(x)
    
    # Couche de sortie
    output = layers.Dense(config.n_classes, activation='softmax', name='output')(x)
    
    # Création du modèle
    model = Model(inputs=input_layer, outputs=output, name='EEG_CNN_LSTM_Temporal')
    
    return model

def normalize_data(X_train, X_val, X_test, config):
    """
    Normalise les données EEG par canal
    """
    print("Normalisation des données...")
    
    X_train_norm = np.zeros_like(X_train)
    X_val_norm = np.zeros_like(X_val)
    X_test_norm = np.zeros_like(X_test)
    
    # Normalisation par canal
    for channel in range(config.n_channels):
        scaler = StandardScaler()
        
        # Fit sur train, transform sur tous
        train_channel_data = X_train[:, :, channel].reshape(-1, 1)
        scaler.fit(train_channel_data)
        
        # Transform
        X_train_norm[:, :, channel] = scaler.transform(
            X_train[:, :, channel].reshape(-1, 1)
        ).reshape(X_train[:, :, channel].shape)
        
        X_val_norm[:, :, channel] = scaler.transform(
            X_val[:, :, channel].reshape(-1, 1)
        ).reshape(X_val[:, :, channel].shape)
        
        X_test_norm[:, :, channel] = scaler.transform(
            X_test[:, :, channel].reshape(-1, 1)
        ).reshape(X_test[:, :, channel].shape)
    
    return X_train_norm, X_val_norm, X_test_norm

def split_data_by_patient(X, y, patients, test_patients=[1, 2], val_patients=[3]):
    """
    Divise les données en s'assurant que les patients ne sont pas mélangés
    entre train/val/test pour éviter le data leakage
    """
    print("Division des données par patient...")
    
    # Indices pour chaque ensemble
    test_idx = np.isin(patients, test_patients)
    val_idx = np.isin(patients, val_patients)
    train_idx = ~(test_idx | val_idx)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    print(f"Patients train: {np.unique(patients[train_idx])}")
    print(f"Patients val: {np.unique(patients[val_idx])}")
    print(f"Patients test: {np.unique(patients[test_idx])}")
    print(f"Tailles - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def compile_model(model, config):
    """
    Compile le modèle avec les paramètres optimisés
    """
    optimizer = Adam(learning_rate=config.learning_rate, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',  # Pas besoin de one-hot encoding
        metrics=['accuracy']
    )
    
    return model

def create_callbacks():
    """
    Crée les callbacks pour l'entraînement
    """
    callbacks = [
        # Réduction du learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Sauvegarde du meilleur modèle
        keras.callbacks.ModelCheckpoint(
            'best_eeg_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks

def plot_training_history(history):
    """
    Visualise l'historique d'entraînement
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='red')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', color='blue')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='red')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Évalue le modèle sur les données de test
    """
    print("Évaluation du modèle...")
    
    # Prédictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Noms des classes
    class_names = label_encoder.classes_
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérité terrain')
    plt.show()
    
    # Accuracy par classe
    accuracy_by_class = cm.diagonal() / cm.sum(axis=1)
    print("\nAccuracy par classe:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {accuracy_by_class[i]:.3f}")

def main(csv_path):
    """
    Fonction principale
    """
    # Configuration
    config = EEGConfig()
    
    # Chargement et préprocessing des données
    X, y, patients, label_encoder, sequence_length = load_and_preprocess_data(csv_path, config)
    
    # Division des données par patient
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_patient(
        X, y, patients, test_patients=[1, 2], val_patients=[3]
    )
    
    # Normalisation
    X_train_norm, X_val_norm, X_test_norm = normalize_data(X_train, X_val, X_test, config)
    
    # Création du modèle
    model = create_cnn_lstm_model(config, sequence_length)
    model = compile_model(model, config)
    
    # Affichage de l'architecture
    print("\nArchitecture du modèle:")
    model.summary()
    
    # Callbacks
    callbacks = create_callbacks()
    
    # Entraînement
    print("\nDébut de l'entraînement...")
    history = model.fit(
        X_train_norm, y_train,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_data=(X_val_norm, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Visualisation de l'entraînement
    plot_training_history(history)
    
    # Évaluation finale
    evaluate_model(model, X_test_norm, y_test, label_encoder)
    
    return model, history, label_encoder

# Exemple d'utilisation
if __name__ == "__main__":

    csv_dir_path = kagglehub.dataset_download("aymanmostafa11/eeg-motor-imagery-bciciv-2a")
    csv_path = csv_dir_path + '/BCICIV_2a_all_patients.csv'
    model, history, label_encoder = main(csv_path)