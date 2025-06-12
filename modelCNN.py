# optimized_cnn_eeg.py - CNN 2D rapide et efficace pour BCI IV 2a
import os
import numpy as np
import mne
from os import listdir
from os.path import isfile, join
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration pour reproductibilité
tf.random.set_seed(42)
np.random.seed(42)

# Configuration globale
DATA_DIR = 'data'
RAW_DIR = join(DATA_DIR, 'raw') 
DATA_PREPROCESSED_DIR = join(DATA_DIR, 'preprocessed') 

class EfficientEEGPreprocessor:
    """
    Préprocesseur optimisé pour CNN 2D avec dimensions contrôlées
    Format final: (n_epochs, height, width, channels) pour CNN 2D
    """
    
    def __init__(self, target_freq=128, time_window=3.0, n_freq_bins=32):
        """
        Args:
            target_freq: Fréquence de rééchantillonnage (128Hz optimal)
            time_window: Durée de la fenêtre (3s pour réduire les dimensions)
            n_freq_bins: Nombre de bins fréquentiels (32 pour contrôler la taille)
        """
        self.target_freq = target_freq
        self.time_window = time_window
        self.n_time_points = int(target_freq * time_window)  # 384 points
        self.n_freq_bins = n_freq_bins
        
    def load_and_format_raw(self, file_path):
        """Charger et formater un fichier EEG avec preprocessing minimal"""
        try:
            raw = mne.io.read_raw_gdf(file_path, eog=[22,23,24], preload=True, verbose=False)
        except:
            return None
        
        # Mapping des canaux pour BCI IV 2a - sélection des 22 canaux EEG
        mapping = {
            'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 
            'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-6': 'C3', 
            'EEG-C3': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 
            'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz',
            'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-Pz': 'Pz', 'EEG-14': 'P1',
            'EEG-15': 'P2', 'EEG-16': 'POz'
        }
        
        raw.rename_channels(mapping)
        
        # Filtrage large pour préserver l'information
        raw.filter(7.0, 35.0, fir_design='firwin', verbose=False)
        
        # Downsampling agressif pour réduire les dimensions
        raw.resample(self.target_freq, verbose=False)
        
        # Référence moyenne
        raw.set_eeg_reference('average', verbose=False)
        
        # Supprimer EOG et canaux non-EEG
        eeg_channels = [ch for ch in raw.ch_names if not any(x in ch for x in ['EOG', 'ECG', 'EMG'])]
        raw.pick_channels(eeg_channels[:22])  # Limiter à 22 canaux max
        
        return raw
    
    def extract_epochs(self, raw):
        """Extraire les époques avec fenêtre optimisée"""
        events, events_ids = mne.events_from_annotations(raw, verbose=False)
        
        # Mapping des événements BCI IV 2a
        motor_event_ids = ['769', '770', '771', '772']  # left, right, feet, tongue
        try:
            motor_event_ids_effective = [events_ids[key] for key in motor_event_ids if key in events_ids]
        except:
            return None, None
        
        if len(motor_event_ids_effective) < 4:
            # Essayer avec d'autres mappings possibles
            alt_mapping = [1, 2, 3, 4]
            motor_event_ids_effective = [v for v in events_ids.values() if v in alt_mapping]
        
        if len(motor_event_ids_effective) < 2:
            return None, None
        
        # Créer le mapping d'événements
        event_mapping = {}
        class_names = ['left_hand', 'right_hand', 'feet', 'tongue']
        for i, event_id in enumerate(motor_event_ids_effective[:4]):
            if i < len(class_names):
                event_mapping[class_names[i]] = event_id
        
        # Fenêtre d'imagerie motrice optimisée (1.5-4.5s après cue)
        tmin = 1.5
        tmax = tmin + self.time_window
        
        epochs = mne.Epochs(
            raw, events, event_mapping,
            tmin, tmax, picks=['eeg'],
            baseline=None,
            preload=True,
            reject=dict(eeg=150e-6),  # Rejet moins strict
            reject_by_annotation=True,
            verbose=False
        )
        
        return epochs, event_mapping
    
    def create_efficient_2d_representation(self, epochs):
        """
        Créer une représentation 2D compacte et efficace
        Format final: (n_epochs, n_channels, n_time_reduced, n_freq_bands)
        """
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2]
        
        # Normaliser les labels pour qu'ils commencent à 0
        unique_labels = np.unique(y)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        y_normalized = np.array([label_mapping[label] for label in y])
        
        n_epochs, n_channels, n_times = X.shape
        print(f"Données brutes: {X.shape}")
        
        # Réduction temporelle par moyennage par blocs
        time_reduction_factor = 4  # Réduire de 384 à ~96 points
        n_time_reduced = n_times // time_reduction_factor
        
        # Bandes de fréquences simplifiées
        freq_bands = [
            (8, 13),   # Alpha/Mu
            (14, 20),  # Beta bas
            (21, 30),  # Beta haut
            (31, 35)   # Gamma bas
        ]
        n_bands = len(freq_bands)
        
        # Format final optimisé: (n_epochs, n_channels, n_time_reduced, n_bands)
        X_2d = np.zeros((n_epochs, n_channels, n_time_reduced, n_bands))
        
        print("Calcul des caractéristiques fréquentielles...")
        
        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                signal_data = X[epoch_idx, ch_idx, :]
                
                # Réduction temporelle simple par moyennage
                signal_reduced = signal_data[:n_time_reduced * time_reduction_factor].reshape(
                    n_time_reduced, time_reduction_factor
                ).mean(axis=1)
                
                # Calcul des caractéristiques fréquentielles avec fenêtre glissante
                for t_idx in range(n_time_reduced):
                    # Fenêtre temporelle centrée
                    t_start = max(0, t_idx * time_reduction_factor - 32)
                    t_end = min(n_times, t_start + 64)
                    window_data = X[epoch_idx, ch_idx, t_start:t_end]
                    
                    if len(window_data) < 32:
                        continue
                    
                    # FFT rapide
                    fft_data = np.fft.fft(window_data)
                    freqs = np.fft.fftfreq(len(window_data), 1/self.target_freq)
                    power_spectrum = np.abs(fft_data)**2
                    
                    # Extraire l'énergie dans chaque bande
                    for band_idx, (f_low, f_high) in enumerate(freq_bands):
                        freq_mask = (freqs >= f_low) & (freqs <= f_high)
                        if np.any(freq_mask):
                            band_power = np.mean(power_spectrum[freq_mask])
                            X_2d[epoch_idx, ch_idx, t_idx, band_idx] = band_power
        
        # Normalisation log pour stabilité numérique
        X_2d = np.log(X_2d + 1e-8)
        
        print(f"Données 2D finales: {X_2d.shape}")
        print(f"Dimensions: {np.prod(X_2d.shape[1:])} paramètres par époque")
        print(f"Distribution des classes: {np.unique(y_normalized, return_counts=True)}")
        
        return X_2d, y_normalized
    
    def preprocess_all_files(self, data_dir):
        """Traiter tous les fichiers .gdf avec gestion d'erreurs robuste"""
        files = [f for f in listdir(data_dir) 
                if isfile(join(data_dir, f)) and f.endswith('.gdf')]
        
        if not files:
            raise FileNotFoundError(f"Aucun fichier .gdf dans {data_dir}")
        
        all_X = []
        all_y = []
        processed_files = 0
        
        for file in files:
            print(f"Traitement: {file}")
            file_path = join(data_dir, file)
            
            try:
                raw = self.load_and_format_raw(file_path)
                if raw is None:
                    print(f"  ❌ Impossible de charger le fichier")
                    continue
                    
                epochs, event_mapping = self.extract_epochs(raw)
                
                if epochs is not None and len(epochs) > 10:  # Au moins 10 époques
                    X_2d, y = self.create_efficient_2d_representation(epochs)
                    all_X.append(X_2d)
                    all_y.append(y)
                    processed_files += 1
                    print(f"  ✅ {len(epochs)} époques extraites")
                else:
                    print(f"  ❌ Pas assez d'époques valides")
                    
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
                continue
        
        if not all_X:
            raise ValueError("Aucune donnée extraite")
        
        print(f"\n✅ {processed_files} fichiers traités avec succès")
        
        # Concaténation finale
        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        
        # Normalisation globale par canal et bande
        X_shape = X_final.shape
        for ch in range(X_shape[1]):
            for band in range(X_shape[3]):
                channel_band_data = X_final[:, ch, :, band]
                if np.std(channel_band_data) > 0:
                    X_final[:, ch, :, band] = (channel_band_data - np.mean(channel_band_data)) / np.std(channel_band_data)
        
        print(f"✅ Données finales: {X_final.shape}")
        print(f"✅ Total époques: {len(X_final)}")
        print(f"✅ Classes: {np.unique(y_final, return_counts=True)}")
        
        return X_final, y_final

class LightweightCNN2D:
    """
    CNN 2D léger et efficace pour EEG - Optimisé pour CPU
    Architecture minimaliste avec moins de 50k paramètres
    """
    
    def __init__(self, input_shape, num_classes=4):
        """
        Args:
            input_shape: (n_channels, n_time_reduced, n_freq_bands)
            num_classes: Nombre de classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        self._build_efficient_model()
    
    def _build_efficient_model(self):
        """
        Architecture CNN 2D ultra-légère et rapide
        Moins de 50k paramètres pour entraînement rapide sur CPU
        """
        model = models.Sequential([
            # Couche d'entrée avec peu de filtres
            layers.Conv2D(8, (3, 2), activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),
            layers.Dropout(0.3),
            
            # Deuxième couche CNN - Focus spatial
            layers.Conv2D(16, (2, 2), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),
            layers.Dropout(0.3),
            
            # Couche de réduction dimensionnelle
            layers.GlobalAveragePooling2D(),  # Très efficace vs Flatten
            
            # Classification finale minimaliste
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Optimiseur avec learning rate adaptatif
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.003, weight_decay=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        # Afficher le nombre de paramètres
        total_params = model.count_params()
        print(f"✅ Modèle CNN léger créé: {total_params:,} paramètres")
        
        if total_params > 100000:
            print("⚠️  Attention: modèle peut-être trop lourd pour entraînement rapide")
        
        return model
    
    def train_fast(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=80, batch_size=64, verbose=1):
        """
        Entraînement optimisé pour vitesse sur CPU
        """
        print("🚀 Entraînement CNN léger...")
        
        # Conversion en categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        
        # Gestion de la validation
        if X_val is not None and y_val is not None:
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            validation_data = (X_val, y_val_cat)
            validation_split = None
        else:
            validation_data = None
            validation_split = 0.15  # Validation split réduit
        
        # Callbacks optimisés pour vitesse
        callback_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.7, patience=6, 
                min_lr=1e-5, verbose=verbose
            ),
            callbacks.EarlyStopping(
                monitor='val_loss', patience=12, 
                restore_best_weights=True, verbose=verbose
            )
        ]
        
        start_time = time.time()
        
        # Entraînement avec batch size élevé pour vitesse
        self.history = self.model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,  # Batch size élevé pour CPU
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=verbose,
        )
        
        training_time = time.time() - start_time
        print(f"⏱️ Temps d'entraînement: {training_time/60:.1f} minutes")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évaluation rapide et complète"""
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Prédictions
        y_pred_proba = self.model.predict(X_test, verbose=0, batch_size=128)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métriques
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"\n📊 PERFORMANCE TEST:")
        print(f"🎯 Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
        print(f"📉 Loss: {test_loss:.4f}")
        
        # Rapport détaillé
        class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue'][:self.num_classes]
        print("\n📊 RAPPORT DE CLASSIFICATION:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        return test_accuracy, y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Graphique d'entraînement compact"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], 'b-', label='Train', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Accuracy Evolution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], 'b-', label='Train', linewidth=2)
        ax2.plot(self.history.history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax2.set_title('Loss Evolution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Sauvegarder le modèle"""
        self.model.save(filepath)
        print(f"💾 Modèle sauvé: {filepath}")

def main_efficient_pipeline():
    """
    Pipeline complet optimisé pour vitesse
    """
    print("🚀 PIPELINE CNN 2D EFFICACE POUR BCI")
    print("=" * 50)
    
    # Étape 1: Prétraitement efficace
    print("📊 ÉTAPE 1: Prétraitement optimisé...")
    preprocessor = EfficientEEGPreprocessor(
        target_freq=128, 
        time_window=3.0,  # Fenêtre plus courte
        n_freq_bins=32    # Moins de bins fréquentiels
    )
    
    try:
        X, y = preprocessor.preprocess_all_files(RAW_DIR)
    except Exception as e:
        print(f"❌ Erreur prétraitement: {e}")
        return None, 0.0
    
    # Vérifier les dimensions finales
    total_features = np.prod(X.shape[1:])
    print(f"📏 Features totales par époque: {total_features:,}")
    
    if total_features > 10000:
        print("⚠️  Attention: dimensions encore élevées, considérer plus de réduction")
    
    # Sauvegarder les données
    if not os.path.exists(DATA_PREPROCESSED_DIR):
        os.makedirs(DATA_PREPROCESSED_DIR)
    
    np.savez_compressed(join(DATA_PREPROCESSED_DIR, 'eeg_efficient_data.npz'), 
                       data=X, labels=y)
    print("💾 Données sauvées (compressées)")
    
    # Étape 2: Split stratifié
    print("\n📊 ÉTAPE 2: Division des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes train: {np.unique(y_train, return_counts=True)}")
    print(f"Classes test: {np.unique(y_test, return_counts=True)}")
    
    # Étape 3: Entraînement CNN léger
    print("\n🤖 ÉTAPE 3: Entraînement CNN léger...")
    
    input_shape = X_train.shape[1:]
    cnn = LightweightCNN2D(input_shape=input_shape, num_classes=len(np.unique(y)))
    
    print("\n📋 Architecture du modèle:")
    cnn.model.summary()
    
    # Entraînement rapide
    history = cnn.train_fast(
        X_train, y_train,
        epochs=200,
        batch_size=64,  # Batch size élevé pour vitesse
        verbose=1
    )
    
    # Étape 4: Évaluation
    print("\n📊 ÉTAPE 4: Évaluation finale...")
    test_accuracy, y_pred, y_pred_proba = cnn.evaluate(X_test, y_test)
    
    # Visualisation
    print("\n📈 ÉTAPE 5: Résultats...")
    cnn.plot_training_history()
    
    # Sauvegarde finale
    model_path = join(DATA_PREPROCESSED_DIR, 'cnn_efficient_model.h5')
    cnn.save_model(model_path)
    
    print(f"\n🏆 RÉSULTAT FINAL:")
    print(f"🎯 Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"📏 Paramètres modèle: {cnn.model.count_params():,}")
    print(f"💾 Modèle sauvé: {model_path}")
    
    return cnn, test_accuracy

def main_loaded_data():
    data = np.load(join(DATA_PREPROCESSED_DIR, 'eeg_efficient_data.npz'))
    X = data['data']
    y = data['labels']
    print("💾 Données chargées")
    
    # Étape 2: Split stratifié
    print("\n📊 ÉTAPE 2: Division des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes train: {np.unique(y_train, return_counts=True)}")
    print(f"Classes test: {np.unique(y_test, return_counts=True)}")
    
    # Étape 3: Entraînement CNN léger
    print("\n🤖 ÉTAPE 3: Entraînement CNN léger...")
    
    input_shape = X_train.shape[1:]
    cnn = LightweightCNN2D(input_shape=input_shape, num_classes=len(np.unique(y)))
    
    print("\n📋 Architecture du modèle:")
    cnn.model.summary()
    
    # Entraînement rapide
    history = cnn.train_fast(
        X_train, y_train,
        epochs=200,
        batch_size=64,  # Batch size élevé pour vitesse
        verbose=1
    )
    
    # Étape 4: Évaluation
    print("\n📊 ÉTAPE 4: Évaluation finale...")
    test_accuracy, y_pred, y_pred_proba = cnn.evaluate(X_test, y_test)
    
    # Visualisation
    print("\n📈 ÉTAPE 5: Résultats...")
    cnn.plot_training_history()
    
    # Sauvegarde finale
    model_path = join(DATA_PREPROCESSED_DIR, 'cnn_efficient_model.h5')
    cnn.save_model(model_path)
    
    print(f"\n🏆 RÉSULTAT FINAL:")
    print(f"🎯 Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
    print(f"📏 Paramètres modèle: {cnn.model.count_params():,}")
    print(f"💾 Modèle sauvé: {model_path}")
    
    return cnn, test_accuracy

if __name__ == "__main__":
    # Lancement du pipeline
    model, accuracy = main_loaded_data()
    
    if model is not None:
        print(f"\n✅ SUCCÈS - Accuracy finale: {accuracy:.4f}")
        print("💡 Le modèle est prêt pour l'utilisation!")
    else:
        print("\n❌ ÉCHEC - Vérifiez vos données d'entrée")