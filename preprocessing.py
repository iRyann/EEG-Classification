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
    
    print(f"Époques extraites: {len(epochs)} (fenêtre: {tmin}-{tmax}s)")
    
    # Vérifier distribution des classes
    y = epochs.events[:, 2]
    unique_events, counts = np.unique(y, return_counts=True)
    for event, count in zip(unique_events, counts):
        print(f"  Classe {event}: {count} essais")
    
    return epochs, event_mapping