# improved_preprocessing.py
import os
import numpy as np
import mne
from os import listdir
from os.path import isfile, join
from scipy import signal
from scipy.signal import hilbert, welch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from mne.time_frequency import tfr_multitaper
import joblib
import warnings
warnings.filterwarnings('ignore')

# Global variables
DATA_DIR = 'data'
RAW_DIR = join(DATA_DIR, 'raw') 
DATA_PREPROCESSED_DIR = join(DATA_DIR, 'preprocessed') 
VERBOSE = True
DBG_PRINT = print if VERBOSE else lambda *args, **kwargs: None 


def load_raw_data(dir_path: str) -> list[mne.io.Raw]:
    """
    Charger les données brutes depuis un répertoire donné.
    Args:
        dir_path (str): Chemin du répertoire contenant les fichiers bruts.
    Returns:
        list[mne.io.Raw]: Liste d'enregistrements MNE Raw contenant les données EEG.
    """
    try:
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.gdf')]
        if not files:
            raise FileNotFoundError(f"Aucun fichier .gdf trouvé dans {dir_path}")
        raw_data = []
        for file in files:
            file_path = join(dir_path, file)
            DBG_PRINT(f"Chargement du fichier: {file_path}")
            raw = mne.io.read_raw_gdf(file_path, eog=[22,23,24], preload = True)
            raw_data.append(raw)
        DBG_PRINT(f"{len(raw_data)} fichiers chargés avec succès.")
        return raw_data
    except Exception as e:
        DBG_PRINT(f"Erreur lors du chargement des données brutes: {e}")
        return []

def format_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Formater les données brutes MNE Raw pour une utilisation ultérieure.
    Cette fonction applique un montage standard, définit les canaux EOG,
    référence les EEG, filtre les données et gère les canaux EOG.
    Args:
        raw (mne.io.Raw): Enregistrement MNE Raw contenant les données EEG.
    Returns:
        mne.io.Raw: L'enregistrement formaté.
    """
    DBG_PRINT("=== FORMATAGE DES DONNÉES BRUTES ===")
    
    montage = mne.channels.make_standard_montage('standard_1020')
    mapping = {
    'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
    'EEG-5': 'C5', 'EEG-6': 'C1', 'EEG-7': 'C2', 'EEG-8': 'C6',
    'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CP2', 'EEG-12': 'CP4',
    'EEG-13': 'P1', 'EEG-14': 'P2', 'EEG-15': 'POz', 'EEG-16': 'Oz',
    'EEG-Fz': 'Fz', 'EEG-C3': 'C3', 'EEG-Cz': 'Cz', 'EEG-C4': 'C4', 'EEG-Pz': 'Pz'
    }

    raw.rename_channels(mapping)
    raw.set_montage(montage, verbose=False)
    raw.set_eeg_reference('average')
    raw.filter(0.5, 40.0, fir_design='firwin', verbose=False) 
    raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])  

    return raw

def extract_epochs(raw : mne.io.Raw) -> tuple:
    """
    Extraire les époques d'un enregistrement MNE Raw avec des paramètres optimisés.
    Args:
        raw (mne.io.Raw): Enregistrement MNE Raw contenant les données EEG.
    Returns:
        tuple: Un tuple contenant les données extraites et le mapping des événements.
    """
    DBG_PRINT("=== EXTRACTION D'ÉPOQUES ===")
    
    events, events_ids = mne.events_from_annotations(raw)
    
    motor_event_ids = ['769', '770', '771', '772']
    try:
        motor_event_ids_effective = [events_ids[key] for key in motor_event_ids]
    except KeyError as e:
        DBG_PRINT(f"Erreur: Événement manquant {e}")
        return None, None
    
    event_mapping = {
        'left_hand': motor_event_ids_effective[0],
        'right_hand': motor_event_ids_effective[1],
        'feet': motor_event_ids_effective[2],
        'tongue': motor_event_ids_effective[3]
    }
    
    tmin = -0.5   # Avant le cue pour baseline
    tmax = 2.0   # Fin de l'imagerie motrice
    baseline = (-0.5, 0)  # Baseline avant le cue

    epochs = mne.Epochs(
        raw, events, event_mapping,
        tmin, tmax, picks=['eeg'],
        baseline=baseline,
        preload=True,
        reject=dict(eeg=75e-6),
        reject_by_annotation=True,
        decim=2,  # Downsampling à 125Hz pour réduire le bruit
        verbose=False
    )

    DBG_PRINT(f"Époques extraites: {len(epochs)} (fenêtre: {tmin}-{tmax}s)")

    # Vérifier distribution des classes
    y = epochs.events[:, 2]
    unique_events, counts = np.unique(y, return_counts=True)
    assert(len(unique_events) == 4)
    for event, count in zip(unique_events, counts):
        DBG_PRINT(f"  Classe {event}: {count} essais")

    return epochs, event_mapping

def compute_multitaper_spectrogram(epochs):
    """Version avec normalisation par ligne de base"""
    freqs = np.logspace(np.log10(8), np.log10(30), 15)
    power = tfr_multitaper(
        epochs, freqs=freqs, n_cycles=freqs/2,
        use_fft=True, return_itc=False,
        decim=1, n_jobs=-1,
        average=False, verbose=False
    )
    
    # Conversion en dB
    power_db = 10 * np.log10(power.data + np.finfo(float).eps)
    
    # Normalisation par ligne de base (par exemple, première seconde)
    baseline_period = slice(0, int(power.times[power.times <= 1.0].shape[0]))
    baseline_mean = np.mean(power_db[:, :, :, baseline_period], axis=-1, keepdims=True)
    
    # Soustraction de la ligne de base
    power_normalized = power_db - baseline_mean
    
    return power_normalized

def create_optimized_tensor(power_data, model_type='cnn_2d'):
    """
    Version corrigée et optimisée du formatage de tenseur.
    
    Args:
        power_data: (n_epochs, n_channels, n_freqs, n_times)
        model_type: Type de modèle cible
    
    Returns:
        np.ndarray: Tenseur formaté optimalement
    """
    if not isinstance(power_data, np.ndarray):
        raise TypeError("power_data doit être un numpy array")
    
    if power_data.ndim != 4:
        raise ValueError(f"Attendu 4D, reçu {power_data.ndim}D")

    if model_type == 'cnn_2d':
        # Format optimal pour CNN 2D : pas de changement nécessaire !
        return power_data
    
    elif model_type == 'lstm':
        # Format pour LSTM : (epochs, time, features)
        n_epochs, n_channels, n_freqs, n_times = power_data.shape
        return power_data.reshape(n_epochs, n_times, n_channels * n_freqs)
    
    elif model_type == 'cnn_3d':
        # Format pour CNN 3D : ajouter dimension channel
        return np.expand_dims(power_data, axis=1)
    
    elif model_type == 'hybrid':
        # Retourner les deux formats pour modèle hybride
        n_epochs, n_channels, n_freqs, n_times = power_data.shape
        return {
            'cnn': power_data,
            'lstm': power_data.reshape(n_epochs, n_times, n_channels * n_freqs)
        }
    
    else:
        raise ValueError(f"Type de modèle non supporté: {model_type}")

def preprocess_data(raw_data):
    """Version corrigée avec préservation des différences inter-classes"""
    all_spectrograms = []
    all_labels = []
    
    # Calculer d'abord tous les spectrogrammes
    for raw in raw_data:
        formatted = format_raw(raw)
        epochs, _ = extract_epochs(formatted)
        
        if epochs is not None:
            spectrogram = compute_multitaper_spectrogram(epochs)
            labels = epochs.events[:, 2] - epochs.events[:, 2].min()
            all_spectrograms.append(spectrogram)
            all_labels.append(labels)
    
    # Concaténer avant normalisation
    x = np.concatenate(all_spectrograms, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return x, y

def save_preprocessed_data(data: np.ndarray, labels: np.ndarray, filename: str) -> None:
    """
    Enregistrer les données prétraitées dans un fichier .npy.
    Args:
        data (np.ndarray): Tenseur 3D contenant les données prétraitées.
        labels (np.ndarray): Labels associés aux données.
        filename (str): Nom du fichier pour enregistrer les données.
    """
    if not os.path.exists(DATA_PREPROCESSED_DIR):
        os.makedirs(DATA_PREPROCESSED_DIR)

    file_path = join(DATA_PREPROCESSED_DIR, filename)
    np.savez(file_path, data=data, labels=labels)

    DBG_PRINT(f"Données prétraitées enregistrées dans {file_path}")

def validate_preprocessing_output(x, y) -> bool:
    """Validation des données prétraitées"""
    print(f"Shape des données: {x.shape}")
    print(f"Shape des labels: {y.shape}")
    print(f"Range des données: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Labels uniques: {np.unique(y, return_counts=True)}")
    # Vérifications critiques
    assert len(x) == len(y), "Mismatch entre données et labels"
    assert np.all(np.isfinite(x)), "Données contiennent NaN/Inf"
    assert set(y) == {0, 1, 2, 3}, f"Labels incorrects: {set(y)}"

    return True
def main(verbose: bool = True) -> None:
    """
    Fonction principale pour exécuter le prétraitement des données EEG.
    Args:
        verbose (bool): Si True, active les messages de débogage.
    """
    global VERBOSE, DBG_PRINT
    VERBOSE = verbose
    DBG_PRINT = print if VERBOSE else lambda *args, **kwargs: None

    raw_data = load_raw_data(RAW_DIR)
    if not raw_data:
        DBG_PRINT("Aucune donnée brute chargée. Fin du programme.")
        return
    
    preprocessed_data, labels = preprocess_data(raw_data)
    if validate_preprocessing_output(preprocessed_data, labels) :
        save_preprocessed_data(preprocessed_data, labels, 'preprocessed_data.npz')

if __name__ == "__main__":
    main(verbose=True)
