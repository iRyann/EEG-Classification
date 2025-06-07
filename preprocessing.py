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
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
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
    eog_channels = [22, 23, 24]

    raw.set_montage(montage, verbose=False)
    raw.set_eog_channels(eog_channels, verbose=False)
    raw.set_eeg_reference('average', projection=True)

    raw.filter(4.0, 40.0, fir_design='firwin', verbose=False)
    raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.drop_channels(eog_channels, verbose=False)
    
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
        motor_event_ids_effective[0]: 0,  # Left hand
        motor_event_ids_effective[1]: 1,  # Right hand  
        motor_event_ids_effective[2]: 2,  # Feet
        motor_event_ids_effective[3]: 3   # Tongue
    }
    
    tmin = 1.0  # 0.5s après le cue
    tmax = 3.0  # 2s d'imagerie motrice active
    baseline = (-1, -0.5)  # Avant le cue pr normaliser les données

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
    for event, count in zip(unique_events, counts):
        DBG_PRINT(f"  Classe {event}: {count} essais")

    return epochs, event_mapping

def compute_multitaper_spectrogram(epochs: mne.Epochs) -> np.ndarray:
    """
    Calculer le spectrogramme multi-taper pour les époques EEG.
    Args:
        epochs (mne.Epochs): Époques MNE contenant les données EEG.
    Returns:
        np.ndarray: Spectrogramme multi-taper (n_epochs, n_channels, n_freqs, n_times).
    """
    DBG_PRINT("=== CALCUL DU SPECTROGRAMME MULTI-TAPER ===")
    
    freqs = np.arange(8, 31, 1)
    power = tfr_multitaper(
        epochs, freqs=freqs, n_cycles=freqs/2,
        use_fft=True, return_itc=False,
        decim=2, n_jobs=1, 
        average=False, verbose=False
    )

    # Normalisation log pour stabiliser la variance
    power_data = np.log10(power.data + 1e-12)

    return power_data  # (n_epochs, n_channels, n_freqs, n_times)


def create_3d_tensor_corrected(power_data : np.ndarray) -> np.ndarray:
    """
    Créer un tenseur 3D à partir des données de puissance pour l'entrée du modèle CNN 3D.
    Args:
        power_data (np.ndarray): Données de puissance (n_epochs, n_channels, n_freqs, n_times).
    Returns:
        np.ndarray: Tenseur 3D formaté pour le modèle CNN 3D (n_epochs, n_freqs, n_times, n_channels, 1).
    """
    DBG_PRINT("=== CRÉATION DU TENSEUR 3D ===")
    
    # Réorganiser les dimensions pour le modèle CNN 3D
    # Format: (n_epochs, n_freqs, n_times, n_channels, 1)
    x_3d = np.transpose(power_data, (0, 2, 3, 1))  # (epochs, freqs, times, channels)

    DBG_PRINT(f"Tenseur 3D créé: {x_3d.shape}")
    return x_3d

def normalize_spectrograms(power_data):
    """Normalisation robuste des spectrogrammes"""
    from sklearn.preprocessing import RobustScaler
    
    original_shape = power_data.shape
    power_flat = power_data.reshape(original_shape[0], -1)
    scaler = RobustScaler()
    power_normalized = scaler.fit_transform(power_flat)
    power_normalized = power_normalized.reshape(original_shape)
    
    return power_normalized, scaler


def preprocess_data(raw_data: list[mne.io.Raw]) -> tuple:
    """Version corrigée avec labels appropriés"""
    DBG_PRINT("=== PRÉTRAITEMENT DES DONNÉES ===")
    
    all_spectrograms = []
    all_labels = []
    
    for raw in raw_data:
        formatted = format_raw(raw)
        epochs, event_mapping = extract_epochs(formatted)
        
        if epochs is not None:
            spectrogram = compute_multitaper_spectrogram(epochs)
            scaled_spectrogram, _ = normalize_spectrograms(spectrogram)
            labels = epochs.events[:, 2]
            all_spectrograms.append(scaled_spectrogram)
            all_labels.append(labels)

    x = np.concatenate(all_spectrograms, axis=0)
    y = np.concatenate(all_labels, axis=0)
    x_formatted = create_3d_tensor_corrected(x)

    return x_formatted, y

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

def validate_preprocessing_output(x, y):
    """Validation des données prétraitées"""
    print(f"Shape des données: {x.shape}")
    print(f"Shape des labels: {y.shape}")
    print(f"Range des données: [{x.min():.3f}, {x.max():.3f}]")
    print(f"Labels uniques: {np.unique(y, return_counts=True)}")
    
    # Vérifications critiques
    assert len(x) == len(y), "Mismatch entre données et labels"
    assert x.ndim == 4, f"Dimensions incorrectes: {x.ndim} au lieu de 4"
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
    if validate_preprocessing_output(preprocessed_data, labels):
        save_preprocessed_data(preprocessed_data, labels, 'preprocessed_data.npz')

if __name__ == "__main__":
    main(verbose=True)
