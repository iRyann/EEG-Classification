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
from mne.decoding import CSP
import warnings
warnings.filterwarnings('ignore')

# Global variables
DATA_DIR = 'data'  # Répertoire des données
RAW_DIR = join(DATA_DIR, 'raw')  # Répertoire des fichiers bruts
DATA_PREPROCESSED_DIR = join(DATA_DIR, 'preprocessed')  # Répertoire des données prétraitées
VERBOSE = True  # Contrôle l'affichage des messages
DBG_PRINT = print if VERBOSE else lambda *args, **kwargs: None  # Redéfinir print si verbose est False


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
            raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
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

    raw.filter(1.0, 40.0, fir_design='firwin', verbose=False)
    raw.interpolate_bads(reset_bads=True, verbose=False)
    raw.drop_channels(eog_channels, verbose=False)
    
    return raw

def extract_epochs(raw : mne.io.Raw) -> tuple:
    """
    Extraire les époques d'un enregistrement MNE Raw avec des paramètres optimisés.
    Args:
        raw (mne.io.Raw): Enregistrement MNE Raw contenant les données EEG.
    Returns:
        tuple: Un tuple contenant les époques extraites et le mapping des événements.
    """
    print("=== EXTRACTION D'ÉPOQUES ===")
    
    events, events_ids = mne.events_from_annotations(raw)
    
    motor_event_ids = ['769', '770', '771', '772']
    try:
        motor_event_ids_effective = [events_ids[key] for key in motor_event_ids]
    except KeyError as e:
        print(f"Erreur: Événement manquant {e}")
        return None, None
    
    event_mapping = {
        motor_event_ids_effective[0]: 0,  # Left hand
        motor_event_ids_effective[1]: 1,  # Right hand  
        motor_event_ids_effective[2]: 2,  # Feet
        motor_event_ids_effective[3]: 3   # Tongue
    }
    
    tmin = 3.0  # 0.5s après le cue
    tmax = 5.0  # 2s d'imagerie motrice active
    baseline = (-1, -0.5)  # Avant le cue

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