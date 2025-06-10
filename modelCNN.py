# cnn_2d_bci_optimized.py - CNN 2D rapide et efficace pour BCI IV 2a
import os
import numpy as np
import mne
from os import listdir
from os.path import isfile, join
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration pour reproductibilité et performance
tf.random.set_seed(42)
np.random.seed(42)

# Configuration globale
DATA_DIR = 'data'
RAW_DIR = join(DATA_DIR, 'raw') 
DATA_PREPROCESSED_DIR = join(DATA_DIR, 'preprocessed') 
VERBOSE = True

class EEG2DPreprocessor:
    """
    Préprocesseur optimisé pour CNN 2D
    Préserve l'information spatiale et temporelle
    """
    
    def __init__(self, target_freq=225, time_window=2.0):
        """
        Args:
            target_freq: Fréquence cible après downsampling (125Hz pour rapidité)
            time_window: Durée de la fenêtre d'analyse (4s)
        """
        self.target_freq = target_freq
        self.time_window = time_window
        self.n_time_points = int(target_freq * time_window)  # 500 points
        
    def load_and_format_raw(self, file_path):
        """Charger et formater un fichier EEG"""
        raw = mne.io.read_raw_gdf(file_path, eog=[22,23,24], preload=True, verbose=False)
        
        # Mapping des canaux pour BCI IV 2a
        mapping = {
            'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 
            'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-6': 'C3', 
            'EEG-C3': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 
            'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz',
            'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-Pz': 'Pz', 'EEG-14': 'P1',
            'EEG-15': 'P2', 'EEG-16': 'POz'
        }
        
        raw.rename_channels(mapping)
        
        # Montage et filtrage optimisé
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, verbose=False)
        
        # Filtrage dans les bandes pertinentes (8-30Hz: mu/beta)
        raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)
        
        # Downsampling pour accélérer l'entraînement
        raw.resample(self.target_freq, verbose=False)
        
        # Référence moyenne
        raw.set_eeg_reference('average', verbose=False)
        
        # Supprimer EOG
        eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch]
        if eog_channels:
            raw.drop_channels(eog_channels)
            
        return raw
    
    def extract_2d_epochs(self, raw):
        """
        Extraire les époques avec préservation de l'information 2D
        """
        events, events_ids = mne.events_from_annotations(raw, verbose=False)
        
        # Mapping des événements BCI IV 2a
        motor_event_ids = ['769', '770', '771', '772']  # left, right, feet, tongue
        try:
            motor_event_ids_effective = [events_ids[key] for key in motor_event_ids]
        except KeyError as e:
            print(f"Événement manquant: {e}")
            return None, None
        
        event_mapping = {
            'left_hand': motor_event_ids_effective[0],
            'right_hand': motor_event_ids_effective[1], 
            'feet': motor_event_ids_effective[2],
            'tongue': motor_event_ids_effective[3]
        }
        
        # Fenêtre d'imagerie motrice (2-6s après cue, mais on prend 4s)
        tmin = 2.0
        tmax = tmin + self.time_window
        
        epochs = mne.Epochs(
            raw, events, event_mapping,
            tmin, tmax, picks=['eeg'],
            baseline=None,
            preload=True,
            reject=dict(eeg=100e-6),
            reject_by_annotation=True,
            verbose=False
        )
        
        return epochs, event_mapping
    
    def create_2d_representation(self, epochs):
        """
        Créer une représentation 2D optimisée pour CNN
        Format: (n_epochs, n_channels, n_time_points, n_freq_bands)
        """
        X = epochs.get_data()  # (n_epochs, n_channels, n_times)
        y = epochs.events[:, 2] - epochs.events[:, 2].min()  # Labels 0,1,2,3
        
        n_epochs, n_channels, n_times = X.shape
        
        # Option 1: Spectrogramme multi-bandes (recommandé)
        X_2d = self.create_multiband_spectrogram(X)
        
        print(f"Données 2D créées: {X_2d.shape}")
        print(f"Distribution des classes: {np.unique(y, return_counts=True)}")
        
        return X_2d, y
    
    def create_multiband_spectrogram(self, X):
        """
        Créer un spectrogramme multi-bandes optimisé
        """
        n_epochs, n_channels, n_times = X.shape
        
        # Bandes de fréquences pertinentes pour imagerie motrice
        freq_bands = [
            (8, 12),   # Mu
            (13, 18),  # Beta bas
            (19, 25),  # Beta moyen
            (26, 30)   # Beta haut
        ]
        
        n_bands = len(freq_bands)
        
        # Calculer la longueur du spectrogramme
        nperseg = min(64, n_times//4)  # Fenêtre adaptative
        noverlap = nperseg // 2
        
        # Test sur un échantillon pour déterminer la taille
        f_test, t_test, Sxx_test = signal.spectrogram(
            X[0, 0, :], fs=self.target_freq, 
            nperseg=nperseg, noverlap=noverlap
        )
        n_freq_bins, n_time_bins = Sxx_test.shape
        
        # Initialiser le tableau 2D
        X_2d = np.zeros((n_epochs, n_channels, n_time_bins, n_bands))
        
        for epoch_idx in range(n_epochs):
            for ch_idx in range(n_channels):
                # Spectrogramme complet
                f, t, Sxx = signal.spectrogram(
                    X[epoch_idx, ch_idx, :], fs=self.target_freq,
                    nperseg=nperseg, noverlap=noverlap
                )
                
                # Extraire l'énergie dans chaque bande
                for band_idx, (fmin, fmax) in enumerate(freq_bands):
                    freq_mask = (f >= fmin) & (f <= fmax)
                    if np.any(freq_mask):
                        # Moyenne de l'énergie dans la bande
                        band_energy = np.mean(Sxx[freq_mask, :], axis=0)
                        X_2d[epoch_idx, ch_idx, :, band_idx] = band_energy
        
        # Normalisation log pour stabilité
        X_2d = np.log(X_2d + 1e-10)
        
        return X_2d
    
    def preprocess_all_files(self, data_dir):
        """Traiter tous les fichiers .gdf"""
        files = [f for f in listdir(data_dir) 
                if isfile(join(data_dir, f)) and f.endswith('.gdf')]
        
        if not files:
            raise FileNotFoundError(f"Aucun fichier .gdf dans {data_dir}")
        
        all_X = []
        all_y = []
        
        for file in files:
            print(f"Traitement: {file}")
            file_path = join(data_dir, file)
            
            try:
                raw = self.load_and_format_raw(file_path)
                epochs, _ = self.extract_2d_epochs(raw)
                
                if epochs is not None and len(epochs) > 0:
                    X_2d, y = self.create_2d_representation(epochs)
                    all_X.append(X_2d)
                    all_y.append(y)
                    print(f"  ✅ {len(epochs)} époques extraites")
                else:
                    print(f"  ❌ Aucune époque extraite")
                    
            except Exception as e:
                print(f"  ❌ Erreur: {e}")
                continue
        
        if not all_X:
            raise ValueError("Aucune donnée extraite")
        
        # Concaténation
        X_final = np.concatenate(all_X, axis=0)
        y_final = np.concatenate(all_y, axis=0)
        
        # Normalisation globale
        scaler = StandardScaler()
        X_shape = X_final.shape
        X_reshaped = X_final.reshape(-1, X_shape[-1])
        X_normalized = scaler.fit_transform(X_reshaped)
        X_final = X_normalized.reshape(X_shape)
        
        print(f"\n✅ Données finales: {X_final.shape}")
        print(f"✅ Labels: {np.unique(y_final, return_counts=True)}")
        
        return X_final, y_final, scaler

class OptimizedCNN2D:
    """
    CNN 2D optimisé pour EEG - Entraînement rapide sans GPU
    """
    
    def __init__(self, input_shape, num_classes=4):
        """
        Args:
            input_shape: (n_channels, n_time_bins, n_freq_bands)
            num_classes: Nombre de classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        self._build_model()
    
    def _build_model(self):
        """
        Architecture CNN 2D optimisée pour rapidité et performance
        """
        model = models.Sequential([
            # Première couche CNN - Filtres spatiaux
            layers.Conv2D(16, (3, 3), activation='relu', 
                         input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 1)),  # Pool seulement sur l'axe temporel
            layers.Dropout(0.25),
            
            # Deuxième couche CNN - Filtres temporels
            layers.Conv2D(32, (1, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((1, 2)),  # Pool sur l'axe fréquentiel
            layers.Dropout(0.25),
            
            # Troisième couche CNN - Combinaison spatio-temporelle
            layers.Conv2D(32, (2, 2), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Plus efficace que Flatten
            layers.Dropout(0.4),
            
            # Couches denses finales - Compactes
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Optimiseur avec learning rate adaptatif
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.002),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(f"✅ Modèle CNN 2D créé: {model.count_params()} paramètres")
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=60, batch_size=32, verbose=1):
        """
        Entraînement rapide avec callbacks optimisés
        """
        print("🚀 Entraînement CNN 2D...")
        
        # Conversion en categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        
        # Validation split ou données externes
        if X_val is not None and y_val is not None:
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            validation_data = (X_val, y_val_cat)
            validation_split = None
        else:
            validation_data = None
            validation_split = 0.2
        
        # Callbacks pour accélérer l'entraînement
        callback_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.6, patience=8, 
                min_lr=1e-6, verbose=verbose
            ),
            callbacks.EarlyStopping(
                monitor='val_loss', patience=15, 
                restore_best_weights=True, verbose=verbose
            )
        ]
        
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callback_list,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"⏱️ Temps d'entraînement: {training_time/60:.1f} minutes")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évaluation complète"""
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Prédictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métriques
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"\n📊 PERFORMANCE TEST CNN 2D:")
        print(f"🎯 Accuracy: {test_accuracy:.3f}")
        print(f"📉 Loss: {test_loss:.3f}")
        
        print("\n📊 RAPPORT DE CLASSIFICATION:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue']))
        
        print("\n🎯 MATRICE DE CONFUSION:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return test_accuracy, y_pred, y_pred_proba
    
    def plot_training_history(self):
        """Graphique d'entraînement"""
        if self.history is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], 'b-', label='Train')
        ax1.plot(self.history.history['val_accuracy'], 'r-', label='Validation')
        ax1.set_title('CNN 2D - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], 'b-', label='Train')
        ax2.plot(self.history.history['val_loss'], 'r-', label='Validation')
        ax2.set_title('CNN 2D - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Sauvegarder le modèle"""
        self.model.save(filepath)
        print(f"💾 Modèle CNN 2D sauvé: {filepath}")

def main_cnn_2d_pipeline():
    """
    Pipeline complet CNN 2D optimisé
    """
    print("🚀 PIPELINE CNN 2D POUR BCI IV 2a")
    print("=" * 50)
    
    # Étape 1: Prétraitement 2D
    print("📊 ÉTAPE 1: Prétraitement 2D...")
    preprocessor = EEG2DPreprocessor(target_freq=125, time_window=4.0)
    
    try:
        X, y, scaler = preprocessor.preprocess_all_files(RAW_DIR)
    except Exception as e:
        print(f"❌ Erreur prétraitement: {e}")
        return None
    
    # Sauvegarder les données prétraitées
    if not os.path.exists(DATA_PREPROCESSED_DIR):
        os.makedirs(DATA_PREPROCESSED_DIR)
    
    np.savez(join(DATA_PREPROCESSED_DIR, 'eeg_2d_data.npz'), 
             data=X, labels=y)
    print("💾 Données 2D sauvées")
    
    # Étape 2: Split des données
    print("\n📊 ÉTAPE 2: Division des données...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Étape 3: Entraînement CNN 2D
    print("\n🤖 ÉTAPE 3: Entraînement CNN 2D...")
    
    input_shape = X_train.shape[1:]  # (n_channels, n_time_bins, n_freq_bands)
    cnn_2d = OptimizedCNN2D(input_shape=input_shape, num_classes=4)
    
    # Afficher l'architecture
    print("\n📋 Architecture du modèle:")
    cnn_2d.model.summary()
    
    # Entraînement
    history = cnn_2d.train(
        X_train, y_train,
        epochs=120,  # Optimisé pour 30-40 minutes
        batch_size=16,  # Batch size plus petit pour stabilité
        verbose=1
    )
    
    # Étape 4: Évaluation
    print("\n📊 ÉTAPE 4: Évaluation...")
    test_accuracy, y_pred, y_pred_proba = cnn_2d.evaluate(X_test, y_test)
    
    # Graphiques
    print("\n📈 ÉTAPE 5: Visualisation...")
    cnn_2d.plot_training_history()
    
    # Sauvegarde
    model_path = join(DATA_PREPROCESSED_DIR, 'cnn_2d_bci_model.h5')
    cnn_2d.save_model(model_path)
    
    print(f"\n🏆 RÉSULTAT FINAL:")
    print(f"🎯 Accuracy CNN 2D: {test_accuracy:.3f}")
    print(f"💾 Modèle sauvé: {model_path}")
    
    return cnn_2d, test_accuracy

def load_and_test_saved_model():
    """
    Charger et tester un modèle sauvé
    """
    try:
        # Charger les données
        data = np.load(join(DATA_PREPROCESSED_DIR, 'eeg_2d_data.npz'))
        X, y = data['data'], data['labels']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Charger le modèle
        model_path = join(DATA_PREPROCESSED_DIR, 'cnn_2d_bci_model.h5')
        loaded_model = keras.models.load_model(model_path)
        
        # Test
        y_test_cat = to_categorical(y_test, num_classes=4)
        test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"🔄 Modèle chargé - Accuracy: {test_accuracy:.3f}")
        
        return loaded_model, test_accuracy
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
        return None, 0.0

if __name__ == "__main__":
    print("🔧 MENU CNN 2D BCI")
    print("1️⃣ Pipeline complet (prétraitement + entraînement)")
    print("2️⃣ Tester modèle sauvé")
    
    choice = input("Choix (1-2, ou Entrée pour pipeline complet): ").strip()
    
    if choice == '2':
        model, accuracy = load_and_test_saved_model()
    else:
        model, accuracy = main_cnn_2d_pipeline()
    
    print(f"\n✅ Terminé - Accuracy finale: {accuracy:.3f}")