# preprocessing_fixed.py - Version corrigée pour BCI IV 2a
import os
import numpy as np
import mne
from os import listdir
from os.path import isfile, join
from scipy import signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Global variables
DATA_DIR = 'data'
RAW_DIR = join(DATA_DIR, 'raw') 
DATA_PREPROCESSED_DIR = join(DATA_DIR, 'preprocessed') 
VERBOSE = True
DBG_PRINT = print if VERBOSE else lambda *args, **kwargs: None 

def load_raw_data(dir_path: str) -> list[mne.io.Raw]:
    """Charger les données brutes - IDENTIQUE"""
    try:
        files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and f.endswith('.gdf')]
        if not files:
            raise FileNotFoundError(f"Aucun fichier .gdf trouvé dans {dir_path}")
        raw_data = []
        for file in files:
            file_path = join(dir_path, file)
            DBG_PRINT(f"Chargement du fichier: {file_path}")
            raw = mne.io.read_raw_gdf(file_path, eog=[22,23,24], preload=True)
            raw_data.append(raw)
        DBG_PRINT(f"{len(raw_data)} fichiers chargés avec succès.")
        return raw_data
    except Exception as e:
        DBG_PRINT(f"Erreur lors du chargement des données brutes: {e}")
        return []

def format_raw(raw: mne.io.Raw) -> mne.io.Raw:
    """Formatage optimisé pour BCI IV 2a"""
    DBG_PRINT("=== FORMATAGE DES DONNÉES BRUTES ===")
    
    # Mapping corrigé pour BCI IV 2a
    mapping = {
        'EEG-Fz': 'Fz', 'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 
        'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-6': 'C3', 
        'EEG-C3': 'C1', 'EEG-Cz': 'Cz', 'EEG-7': 'C2', 'EEG-C4': 'C4', 
        'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz',
        'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-Pz': 'Pz', 'EEG-14': 'P1',
        'EEG-15': 'P2', 'EEG-16': 'POz'
    }
    
    raw.rename_channels(mapping)
    
    # Montage standard
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, verbose=False)
    
    # Filtrage optimisé pour imagerie motrice
    raw.filter(8.0, 30.0, fir_design='firwin', verbose=False)  # Bandes mu/beta
    
    # Référence moyenne
    raw.set_eeg_reference('average', verbose=False)
    
    # Supprimer EOG
    if 'EOG-left' in raw.ch_names:
        raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    
    return raw

def extract_epochs_corrected(raw: mne.io.Raw) -> tuple:
    """
    ✅ CORRECTION MAJEURE : Fenêtre temporelle correcte pour BCI IV 2a
    """
    DBG_PRINT("=== EXTRACTION D'ÉPOQUES CORRIGÉE ===")
    
    events, events_ids = mne.events_from_annotations(raw)
    
    # Mapping correct des événements BCI IV 2a
    motor_event_ids = ['769', '770', '771', '772']  # left, right, feet, tongue
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
    
    # ✅ CORRECTION CRUCIALE : Fenêtre correcte pour BCI IV 2a
    tmin = 2.0    # Début de l'imagerie motrice (2s après cue)
    tmax = 6.0    # Fin de l'imagerie motrice (6s après cue)
    baseline = None  # Pas de baseline automatique
    
    epochs = mne.Epochs(
        raw, events, event_mapping,
        tmin, tmax, picks=['eeg'],
        baseline=baseline,
        preload=True,
        reject=dict(eeg=100e-6),  # Rejet plus permissif
        reject_by_annotation=True,
        decim=1,  # Garder 250Hz pour préserver l'information
        verbose=False
    )
    
    DBG_PRINT(f"Époques extraites: {len(epochs)} (fenêtre: {tmin}-{tmax}s)")
    
    # Vérifier distribution des classes
    y = epochs.events[:, 2]
    unique_events, counts = np.unique(y, return_counts=True)
    for event, count in zip(unique_events, counts):
        DBG_PRINT(f"  Classe {event}: {count} essais")
    
    return epochs, event_mapping

def compute_csp_features(epochs, n_components=6):
    """
    ✅ SOLUTION : CSP au lieu de spectrogramme multitaper
    CSP est LA méthode de référence pour BCI moteur
    """
    from mne.decoding import CSP
    
    DBG_PRINT("=== EXTRACTION FEATURES CSP ===")
    
    # Préparer les données
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2] - epochs.events[:, 2].min()  # Labels 0,1,2,3
    
    # ✅ CORRECTION : log=None avec transform_into='csp_space'
    csp = CSP(n_components=n_components, reg='ledoit_wolf', log=None, 
              cov_est='concat', transform_into='csp_space')
    
    # Extraction des features CSP
    X_csp = csp.fit_transform(X, y)
    
    # ✅ Application manuelle du log si nécessaire
    # Le log améliore souvent les performances pour la classification
    X_csp_log = np.log(np.abs(X_csp) + 1e-8)
    
    DBG_PRINT(f"Features CSP: {X_csp_log.shape}")
    DBG_PRINT(f"Range CSP: [{X_csp_log.min():.3f}, {X_csp_log.max():.3f}]")
    
    return X_csp_log, csp

def compute_csp_features_simple(epochs, n_components=6):
    """
    Version ultra-simple du CSP sans complications
    """
    from mne.decoding import CSP
    
    DBG_PRINT("=== EXTRACTION FEATURES CSP SIMPLE ===")
    
    # Préparer les données
    X = epochs.get_data()  # (n_epochs, n_channels, n_times)
    y = epochs.events[:, 2] - epochs.events[:, 2].min()  # Labels 0,1,2,3
    
    # CSP ultra-simple - configuration minimale qui marche toujours
    csp = CSP(n_components=n_components, reg=None, log=True)
    
    # Extraction directe
    X_csp = csp.fit_transform(X, y)
    
    DBG_PRINT(f"Features CSP: {X_csp.shape}")
    DBG_PRINT(f"Range CSP: [{X_csp.min():.3f}, {X_csp.max():.3f}]")
    
    return X_csp, csp
    """
    Alternative : PSD dans les bandes de fréquences clés
    """
    DBG_PRINT("=== EXTRACTION FEATURES PSD ===")
    
    X = epochs.get_data()  # (epochs, channels, time)
    
    # Bandes de fréquences pour imagerie motrice
    freq_bands = {
        'mu': (8, 12),
        'beta': (13, 30)
    }
    
    features = []
    for epoch in X:
        epoch_features = []
        for ch_idx in range(epoch.shape[0]):
            ch_data = epoch[ch_idx, :]
            
            for band_name, (fmin, fmax) in freq_bands.items():
                # PSD dans la bande
                freqs, psd = signal.welch(ch_data, fs=250, nperseg=128)
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                power = np.mean(psd[freq_mask])
                epoch_features.append(power)
        
        features.append(epoch_features)
    
    features = np.array(features)
    DBG_PRINT(f"Features PSD: {features.shape}")
    
    return features

def preprocess_data_corrected(raw_data, method='csp'):
    """
    ✅ Pipeline de prétraitement corrigé
    """
    all_features = []
    all_labels = []
    
    for raw in raw_data:
        formatted = format_raw(raw)
        epochs, _ = extract_epochs_corrected(formatted)
        
        if epochs is not None and len(epochs) > 0:
            if method == 'csp':
                features, _ = compute_csp_features(epochs)
            elif method == 'csp_simple':
                features, _ = compute_csp_features_simple(epochs)
            elif method == 'psd':
                features = compute_power_spectral_density(epochs)
            else:
                raise ValueError(f"Méthode inconnue: {method}")
            
            labels = epochs.events[:, 2] - epochs.events[:, 2].min()
            all_features.append(features)
            all_labels.append(labels)
    
    # Concaténation
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # ✅ Normalisation légère et appropriée
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    DBG_PRINT(f"Données finales: {X_scaled.shape}")
    DBG_PRINT(f"Labels: {np.unique(y, return_counts=True)}")
    
    return X_scaled, y, scaler

def validate_preprocessing_output_fixed(X, y):
    """Validation rapide sans PCA bloquante"""
    print(f"✅ Shape des données: {X.shape}")
    print(f"✅ Shape des labels: {y.shape}")
    print(f"✅ Range des données: [{X.min():.3f}, {X.max():.3f}]")
    print(f"✅ Labels uniques: {np.unique(y, return_counts=True)}")
    
    # Vérifications critiques
    assert len(X) == len(y), "Mismatch entre données et labels"
    assert np.all(np.isfinite(X)), "Données contiennent NaN/Inf"
    assert set(y) == {0, 1, 2, 3}, f"Labels incorrects: {set(y)}"
    
    # Test de séparabilité rapide
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        score = lda.score(X, y)
        print(f"✅ Score LDA: {score:.3f}")
        
        if score > 0.3:
            print("✅ Données semblent exploitables pour l'apprentissage")
        else:
            print("⚠️ Score LDA faible - vérifier la qualité des données")
    except Exception as e:
        print(f"❌ Erreur LDA: {e}")
    
    return True

def save_preprocessed_data(data: np.ndarray, labels: np.ndarray, scaler, filename: str):
    """Sauvegarder avec le scaler"""
    if not os.path.exists(DATA_PREPROCESSED_DIR):
        os.makedirs(DATA_PREPROCESSED_DIR)
    
    file_path = join(DATA_PREPROCESSED_DIR, filename)
    np.savez(file_path, data=data, labels=labels)
    
    # Sauvegarder le scaler séparément
    import joblib
    scaler_path = join(DATA_PREPROCESSED_DIR, filename.replace('.npz', '_scaler.pkl'))
    joblib.dump(scaler, scaler_path)
    
    DBG_PRINT(f"✅ Données prétraitées sauvées: {file_path}")
    DBG_PRINT(f"✅ Scaler sauvé: {scaler_path}")

def main(verbose: bool = True, method: str = 'csp'):
    """
    Fonction principale corrigée
    """
    global VERBOSE, DBG_PRINT
    VERBOSE = verbose
    DBG_PRINT = print if VERBOSE else lambda *args, **kwargs: None
    
    print("🚀 PREPROCESSING BCI IV 2a - VERSION CORRIGÉE")
    print(f"📊 Méthode d'extraction: {method.upper()}")
    
    raw_data = load_raw_data(RAW_DIR)
    if not raw_data:
        DBG_PRINT("❌ Aucune donnée brute chargée. Fin du programme.")
        return
    
    X, y, scaler = preprocess_data_corrected(raw_data, method=method)
    
    if validate_preprocessing_output_fixed(X, y):
        filename = f'preprocessed_data_{method}.npz'
        save_preprocessed_data(X, y, scaler, filename)
        print(f"✅ Prétraitement terminé avec succès!")
        return X, y, scaler
    else:
        print("❌ Validation échouée")
        return None, None, None

if __name__ == "__main__":
    # Test avec CSP simple (le plus stable)
    X_csp, y_csp, scaler_csp = main(verbose=True, method='csp_simple')
    
    # Alternative avec CSP avancé
    # X_csp, y_csp, scaler_csp = main(verbose=True, method='csp')
    
    # Alternative avec PSD
    # X_psd, y_psd, scaler_psd = main(verbose=True, method='psd')