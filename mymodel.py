# diagnostic_and_fixes.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, Dropout, Flatten, LSTM, Reshape, Concatenate, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2

def diagnostic_data_quality(X, y, class_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue']):
    """
    Diagnostic complet de la qualité des données
    """
    print("=== DIAGNOSTIC DES DONNÉES ===")
    
    # 1. Distribution des classes
    print(f"Distribution des classes: {np.bincount(y)}")
    class_distribution = np.bincount(y) / len(y)
    print(f"Pourcentages: {class_distribution * 100}")
    
    # 2. Statistiques des features
    print(f"\nForme des données: {X.shape}")
    print(f"Min: {X.min():.6f}, Max: {X.max():.6f}")
    print(f"Mean: {X.mean():.6f}, Std: {X.std():.6f}")
    
    # 3. Visualisation des spectrogrammes moyens par classe
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for class_idx in range(4):
        class_mask = y == class_idx
        if np.sum(class_mask) > 0:
            # Moyenne sur tous les essais et canaux pour cette classe
            class_data = X[class_mask].mean(axis=(0, 3, 4))  # Shape: (freq, time)
            
            im = axes[class_idx].imshow(class_data, aspect='auto', origin='lower', 
                                      cmap='viridis', extent=[0, 30, 8, 30])
            axes[class_idx].set_title(f'{class_names[class_idx]} (n={np.sum(class_mask)})')
            axes[class_idx].set_xlabel('Temps (bins)')
            axes[class_idx].set_ylabel('Fréquence (Hz)')
            plt.colorbar(im, ax=axes[class_idx])
    
    plt.tight_layout()
    plt.savefig('spectrograms_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Analyse de la variance inter-classes
    print("\n=== ANALYSE DE VARIANCE INTER-CLASSES ===")
    
    # Calculer la variance intra et inter-classes
    X_flattened = X.reshape(X.shape[0], -1)  # Aplatir toutes les dimensions
    
    total_variance = np.var(X_flattened, axis=0)
    within_class_variances = []
    
    for class_idx in range(4):
        class_mask = y == class_idx
        if np.sum(class_mask) > 1:
            class_var = np.var(X_flattened[class_mask], axis=0)
            within_class_variances.append(class_var)
    
    within_class_variance = np.mean(within_class_variances, axis=0)
    between_class_variance = total_variance - within_class_variance
    
    # Ratio de Fisher (between/within variance)
    fisher_ratio = between_class_variance / (within_class_variance + 1e-8)
    
    print(f"Variance totale moyenne: {total_variance.mean():.6f}")
    print(f"Variance intra-classe moyenne: {within_class_variance.mean():.6f}")
    print(f"Variance inter-classe moyenne: {between_class_variance.mean():.6f}")
    print(f"Ratio de Fisher moyen: {fisher_ratio.mean():.6f}")
    print(f"% de features avec Fisher > 1: {(fisher_ratio > 1).mean() * 100:.1f}%")
    
    return fisher_ratio

def reduce_dimensionality(X_train, X_val, y_train, method='channel_freq_selection', n_components=None):
    """
    Réduction de dimensionnalité intelligente pour les données EEG
    """
    print(f"\n=== RÉDUCTION DE DIMENSIONNALITÉ ({method}) ===")
    
    if method == 'channel_freq_selection':
        # Sélection des canaux et fréquences les plus discriminants
        return select_best_channels_freqs(X_train, X_val, y_train)
    
    elif method == 'pca_per_channel':
        return pca_per_channel(X_train, X_val, n_components or 10)
    
    elif method == 'spatial_filtering':
        return apply_csp_like_filtering(X_train, X_val, y_train)
    
    else:
        return X_train, X_val

def select_best_channels_freqs(X_train, X_val, y_train, n_channels=10, n_freqs=15):
    """
    Sélectionner les meilleurs canaux et fréquences basés sur le score F
    """
    print(f"Sélection des {n_channels} meilleurs canaux et {n_freqs} meilleures fréquences...")
    
    n_trials, n_freq_orig, n_time, n_chan_orig, _ = X_train.shape
    
    # 1. Sélection des canaux
    # Moyenner sur temps et fréquences pour chaque canal
    channel_features = X_train.mean(axis=(1, 2, 4))  # Shape: (trials, channels)
    
    selector_channels = SelectKBest(f_classif, k=n_channels)
    selector_channels.fit(channel_features, y_train)
    selected_channels = selector_channels.get_support()
    
    print(f"Canaux sélectionnés: {np.where(selected_channels)[0]}")
    
    # 2. Sélection des fréquences
    # Moyenner sur temps et canaux pour chaque fréquence
    freq_features = X_train.mean(axis=(2, 3, 4))  # Shape: (trials, freqs)
    
    selector_freqs = SelectKBest(f_classif, k=n_freqs)
    selector_freqs.fit(freq_features, y_train)
    selected_freqs = selector_freqs.get_support()
    
    print(f"Fréquences sélectionnées: {np.where(selected_freqs)[0]}")
    
    # 3. Appliquer la sélection
    X_train_reduced = X_train[:, selected_freqs, :, :, :]
    X_train_reduced = X_train_reduced[:, :, :, selected_channels, :]
    
    X_val_reduced = X_val[:, selected_freqs, :, :, :]
    X_val_reduced = X_val_reduced[:, :, :, selected_channels, :]
    
    print(f"Forme réduite: {X_train_reduced.shape}")
    
    return X_train_reduced, X_val_reduced

def create_improved_3dclmi_model(input_shape, n_classes=4):
    """
    Modèle 3D-CLMI amélioré avec plusieurs corrections
    """
    print(f"Création du modèle amélioré avec input_shape: {input_shape}")
    
    inputs = Input(shape=input_shape)
    
    # === BRANCHE CNN 3D AMÉLIORÉE ===
    # Block 1
    conv1 = Conv3D(16, kernel_size=(3, 3, 1), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001))(inputs)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    
    # Block 2
    conv2 = Conv3D(32, kernel_size=(3, 3, 1), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001))(pool1)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    
    # Block 3
    conv3 = Conv3D(64, kernel_size=(3, 3, 1), activation='relu', padding='same',
                   kernel_regularizer=l2(0.001))(pool2)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    
    # Flatten CNN output
    cnn_output = Flatten()(pool3)
    
    # === BRANCHE LSTM RÉPARÉE ===
    n_freqs, n_times, n_channels, _ = input_shape
    
    # Reshape pour LSTM: traiter chaque fréquence comme un pas de temps
    # Input: (batch, freq, time, channels, 1) -> (batch, freq, time*channels)
    lstm_input = Reshape((n_freqs, n_times * n_channels))(inputs)
    
    # LSTM layers avec dropout
    lstm1 = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(lstm_input)
    lstm2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(lstm1)
    
    # === FUSION ET CLASSIFICATION ===
    merged = Concatenate()([cnn_output, lstm2])
    
    # Fully connected layers avec regularization
    dense1 = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(merged)
    dense1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.5)(dense1)
    
    dense2 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(dropout1)
    dense2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.3)(dense2)
    
    # Output layer
    outputs = Dense(n_classes, activation='softmax')(dropout2)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile avec class weights pour gérer le déséquilibre
    model.compile(
        optimizer=Adam(learning_rate=0.0001),  # Learning rate plus faible
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def calculate_class_weights(y_train):
    """
    Calculer les poids de classe pour gérer le déséquilibre
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Poids de classe calculés: {class_weight_dict}")
    return class_weight_dict

def train_improved_model(model, X_train, y_train, X_val, y_val, batch_size=16, epochs=100):
    """
    Entraînement amélioré avec gestion du déséquilibre
    """
    # Calculer les poids de classe
    class_weights = calculate_class_weights(y_train)
    
    # Callbacks améliorés
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
        ModelCheckpoint('best_improved_3dclmi_model.h5', save_best_only=True, 
                       monitor='val_accuracy', mode='max')
    ]
    
    # Entraînement
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history

def enhanced_evaluation(model, X_test, y_test, class_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue']):
    """
    Évaluation améliorée avec plus de métriques
    """
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
    
    # Prédictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n=== RÉSULTATS D'ÉVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies classes')
    plt.title('Matrice de Confusion - Modèle Amélioré')
    plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Rapport détaillé
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    print(f"\n=== RAPPORT PAR CLASSE ===")
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:12s}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions_proba': y_pred_proba
    }

def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')

# Script principal de correction
def main_fix_model():
    """
    Script principal pour corriger et améliorer le modèle
    """
    print("=== CHARGEMENT DES DONNÉES ===")
    try:
        DATA_PATH = 'data/preprocessed/'  # Chemin des données
        # Supposons que les données sont déjà préprocessées
        X_train = np.load(DATA_PATH + 'X_train_3DCLMI.npy')
        y_train = np.load(DATA_PATH + 'y_train_3DCLMI.npy')
        X_val = np.load(DATA_PATH + 'X_val_3DCLMI.npy')
        y_val = np.load(DATA_PATH + 'y_val_3DCLMI.npy')

        print(f"Données chargées: X_train={X_train.shape}, y_train={y_train.shape}")
        
        # 1. DIAGNOSTIC
        fisher_ratio = diagnostic_data_quality(X_train, y_train)
        
        # 2. RÉDUCTION DE DIMENSIONNALITÉ
        X_train_reduced, X_val_reduced = reduce_dimensionality(
            X_train, X_val, y_train, method='channel_freq_selection'
        )
        
        # 3. CRÉATION DU MODÈLE AMÉLIORÉ
        input_shape = X_train_reduced.shape[1:]
        model = create_improved_3dclmi_model(input_shape)
        print(model.summary())
        
        # 4. ENTRAÎNEMENT
        print("\n=== ENTRAÎNEMENT DU MODÈLE AMÉLIORÉ ===")
        history = train_improved_model(model, X_train_reduced, y_train, 
                                     X_val_reduced, y_val, batch_size=16)

        plot_training_history(history)
        print("Entraînement terminé!")
        
        # 5. ÉVALUATION
        metrics = enhanced_evaluation(model, X_val_reduced, y_val)
        
        # 6. SAUVEGARDE
        model.save('final_improved_3dclmi_model.h5')
        print("\nModèle amélioré sauvegardé!")
        
        return model, history, metrics
        
    except FileNotFoundError as e:
        print(f"Erreur: Fichiers de données non trouvés - {e}")
        print("Assurez-vous d'avoir exécuté le preprocessing d'abord.")
        return None, None, None

if __name__ == "__main__":
    main_fix_model()