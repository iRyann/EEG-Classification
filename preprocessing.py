import os
import numpy as np
import mne
from os import listdir
from os.path import isfile, join
from scipy import signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from mne.time_frequency import morlet

def preprocess_eeg_3dclmi(raw, event_id=None, fs=250):
    """
    Préprocessing EEG optimisé pour le modèle 3D-CLMI
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Données EEG brutes
    event_id : dict
        Dictionnaire des événements
    fs : int
        Fréquence d'échantillonnage
        
    Returns:
    --------
    X : numpy.ndarray
        Données préprocessées de forme (n_trials, 30, 30, 22, 1)
    y : numpy.ndarray
        Labels de forme (n_trials,)
    """
    
    # Extraire les événements
    events, events_ids = mne.events_from_annotations(raw)
    
    # Codes des événements pour les 4 classes d'imagerie motrice
    motor_event_ids = ['769', '770', '771', '772']
    motor_event_ids_effectives = [events_ids[keys] for keys in motor_event_ids]
    
    # Mappage des événements
    event_id = {
        'left_hand': motor_event_ids_effectives[0],
        'right_hand': motor_event_ids_effectives[1],
        'feet': motor_event_ids_effectives[2],
        'tongue': motor_event_ids_effectives[3]
    }
    
    # Filtrage passe-bande (8-30 Hz) pour capturer les rythmes Alpha et Beta
    raw_filtered = raw.copy()
    raw_filtered.filter(8, 30, method='iir')
    
    # Application de la référence moyenne commune (CAR)
    raw_filtered.set_eeg_reference('average')
    
    # Extraction des époques
    # Commencer 2.5s après le cue (cue à t=2s, donc 4.5s depuis le début)
    # Finir 4.5s après le cue (6.5s depuis le début)
    tmin = 4.5  # 2.5s après le cue
    tmax = 6.5  # 4.5s après le cue
    
    # Créer les époques
    epochs = mne.Epochs(raw_filtered, events, event_id, tmin, tmax, 
                        picks=['eeg'], baseline=None, preload=True)
    
    # Obtenir les données et labels
    X = epochs.get_data()  # Forme: (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]
    
    # Mapper les codes d'événements vers des classes indexées à 0
    class_mapping = {
        event_id['left_hand']: 0,
        event_id['right_hand']: 1,
        event_id['feet']: 2,
        event_id['tongue']: 3
    }
    
    y = np.array([class_mapping[code] for code in y])
    
    # Transformer les données dans le format requis par le modèle 3D-CLMI
    X_transformed = transform_to_timefreq_3d(X, fs=fs)
    
    return X_transformed, y

def transform_to_timefreq_3d(X, fs=250, n_freqs=30, n_times=30):
    """
    Transformer les données EEG en représentation temps-fréquence 3D
    optimisée pour 3D-CLMI
    
    Parameters:
    -----------
    X : numpy.ndarray
        Données d'époques EEG de forme (n_trials, n_channels, n_times)
    fs : int
        Fréquence d'échantillonnage
    n_freqs : int
        Nombre de bins de fréquence (30 pour correspondre au modèle)
    n_times : int
        Nombre de bins de temps (30 pour correspondre au modèle)
        
    Returns:
    --------
    X_transformed : numpy.ndarray
        Données transformées de forme (n_trials, 30, 30, n_channels, 1)
    """
    n_trials, n_channels, n_samples = X.shape
    
    # Définir les fréquences d'intérêt (8-30 Hz divisé en 30 bins)
    freqs = np.linspace(8, 30, n_freqs)
    
    # Définir les instants temporels (diviser la période en 30 bins)
    times = np.linspace(0, n_samples/fs, n_times)
    
    # Initialiser le tableau de sortie
    X_tf = np.zeros((n_trials, n_freqs, n_times, n_channels))
    
    print(f"Calcul de la représentation temps-fréquence...")
    
    for trial in range(n_trials):
        if trial % 50 == 0:
            print(f"  Traitement de l'essai {trial+1}/{n_trials}")
            
        for ch in range(n_channels):
            # Obtenir le signal pour ce canal et cet essai
            signal_data = X[trial, ch, :]
            
            # Calculer la représentation temps-fréquence
            X_tf[trial, :, :, ch] = compute_time_frequency_representation(
                signal_data, freqs, times, fs
            )
    
    # Ajouter la dimension des canaux pour CNN: (n_trials, freqs, times, channels, 1)
    X_transformed = X_tf.reshape(n_trials, n_freqs, n_times, n_channels, 1)
    
    print(f"Transformation terminée. Forme finale: {X_transformed.shape}")
    
    return X_transformed

def compute_time_frequency_representation(signal_data, freqs, times, fs):
    """
    Calculer la représentation temps-fréquence d'un signal
    
    Parameters:
    -----------
    signal_data : numpy.ndarray
        Signal temporal (n_samples,)
    freqs : numpy.ndarray
        Fréquences d'intérêt
    times : numpy.ndarray
        Instants temporels d'intérêt
    fs : int
        Fréquence d'échantillonnage
        
    Returns:
    --------
    tf_repr : numpy.ndarray
        Représentation temps-fréquence (n_freqs, n_times)
    """
    n_freqs = len(freqs)
    n_times = len(times)
    tf_repr = np.zeros((n_freqs, n_times))
    
    # Calculer l'indice temporel pour chaque instant d'intérêt
    time_indices = np.round(times * fs).astype(int)
    time_indices = np.clip(time_indices, 0, len(signal_data) - 1)
    
    for f_idx, freq in enumerate(freqs):
        # Méthode 1: Filtrage passe-bande + enveloppe
        # Créer un filtre passe-bande centré sur la fréquence
        bandwidth = 2.0  # Largeur de bande en Hz
        low_freq = max(freq - bandwidth/2, 0.5)
        high_freq = min(freq + bandwidth/2, fs/2 - 0.5)
        
        # Conception du filtre
        b, a = signal.butter(4, [low_freq, high_freq], btype='band', fs=fs)
        
        # Appliquer le filtre
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        # Calculer l'enveloppe using Hilbert transform
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
        # Échantillonner l'enveloppe aux instants d'intérêt
        tf_repr[f_idx, :] = envelope[time_indices]
    
    return tf_repr

def apply_spatial_filtering(X, method='csp', n_components=6):
    """
    Appliquer un filtrage spatial (optionnel pour améliorer les performances)
    
    Parameters:
    -----------
    X : numpy.ndarray
        Données de forme (n_trials, n_freqs, n_times, n_channels, 1)
    method : str
        Méthode de filtrage spatial ('csp', 'ica', etc.)
    n_components : int
        Nombre de composantes à conserver
        
    Returns:
    --------
    X_filtered : numpy.ndarray
        Données filtrées spatialement
    """
    # Cette fonction peut être implémentée selon les besoins
    # Pour l'instant, on retourne les données originales
    return X

def normalize_data(X_train, X_val):
    """
    Normaliser les données (standardisation par canal et par fréquence)
    
    Parameters:
    -----------
    X_train, X_val : numpy.ndarray
        Données d'entraînement et de validation
        
    Returns:
    --------
    X_train_norm, X_val_norm : numpy.ndarray
        Données normalisées
    """
    # Calculer les statistiques sur l'ensemble d'entraînement
    mean_train = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
    std_train = np.std(X_train, axis=(0, 1, 2), keepdims=True)
    
    # Appliquer la normalisation
    X_train_norm = (X_train - mean_train) / (std_train + 1e-8)
    X_val_norm = (X_val - mean_train) / (std_train + 1e-8)
    
    return X_train_norm, X_val_norm

# Script principal de préprocessing
def main_preprocessing():
    """
    Script principal pour le préprocessing des données BCI IV 2a
    """
    print(f"Version {np.__version__}")
    # Définir le chemin vers les fichiers GDF
    GDF_FILES_PATH = '/home/cytech/dev/EEG Classification/data/raw'
    raw_gdfs = []
    
    # Lire tous les fichiers GDF
    gdf_files = [join(GDF_FILES_PATH, f) for f in listdir(GDF_FILES_PATH) 
                 if isfile(join(GDF_FILES_PATH, f)) and f.endswith('.gdf')]
    
    for gdf_file in gdf_files:
        print(f"Chargement de {gdf_file}")
        try:
            raw = mne.io.read_raw_gdf(gdf_file, eog=[22, 23, 24], preload=True)
            raw_gdfs.append(raw)
        except Exception as e:
            print(f"Erreur lors du chargement de {gdf_file}: {e}")
    
    # Traiter chaque fichier GDF
    processed_data = []
    for i, raw in enumerate(raw_gdfs):
        print(f"Traitement du fichier {i+1}/{len(raw_gdfs)}")
        try:
            X, y = preprocess_eeg_3dclmi(raw)
            processed_data.append((X, y))
            print(f"  Extrait {len(y)} essais avec forme {X.shape}")
        except Exception as e:
            print(f"  Erreur lors du traitement du fichier: {e}")
    
    if not processed_data:
        print("Aucune donnée traitée avec succès!")
        return
    
    # Combiner toutes les données traitées
    X_all = np.concatenate([data[0] for data in processed_data], axis=0)
    y_all = np.concatenate([data[1] for data in processed_data], axis=0)
    
    print(f"Forme finale du jeu de données: {X_all.shape}, Forme des labels: {y_all.shape}")
    print(f"Distribution des classes: {np.bincount(y_all)}")
    
    # Diviser en ensembles d'entraînement et de validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Normaliser les données
    X_train_norm, X_val_norm = normalize_data(X_train, X_val)
    
    # Sauvegarder les données préprocessées
    print("Sauvegarde des données préprocessées...")
    NPY_FILES_PATH = '/home/cytech/dev/EEG Classification/data/preprocessed'
    if not os.path.exists(NPY_FILES_PATH):
        os.makedirs(NPY_FILES_PATH)

    # Enregistrer les données prétraitées
    np.save(NPY_FILES_PATH + '/X_train_3DCLMI.npy', X_train_norm)
    np.save(NPY_FILES_PATH + '/y_train_3DCLMI.npy', y_train)
    np.save(NPY_FILES_PATH + '/X_val_3DCLMI.npy', X_val_norm)
    np.save(NPY_FILES_PATH + '/y_val_3DCLMI.npy', y_val)

    print("Préprocessing terminé et données sauvegardées!")
    print(f"Formes finales:")
    print(f"  X_train: {X_train_norm.shape}")
    print(f"  X_val: {X_val_norm.shape}")
    print(f"  y_train: {y_train.shape} - Distribution: {np.bincount(y_train)}")
    print(f"  y_val: {y_val.shape} - Distribution: {np.bincount(y_val)}")

if __name__ == "__main__":
    main_preprocessing()