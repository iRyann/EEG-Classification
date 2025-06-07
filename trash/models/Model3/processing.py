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

def preprocess_eeg_3dclmi_improved(raw, event_id=None, fs=250):
    """
    Preprocessing EEG amélioré pour le modèle 3D-CLMI
    
    Améliorations:
    - Filtrage adaptatif par bande de fréquence
    - Normalisation robuste
    - Vérification de la qualité du signal
    - Extraction d'époques plus robuste
    """
    print("Début du preprocessing amélioré...")
    
    # Extraire les événements
    events, events_ids = mne.events_from_annotations(raw)
    
    # Codes des événements pour les 4 classes d'imagerie motrice
    motor_event_ids = ['769', '770', '771', '772']
    try:
        motor_event_ids_effectives = [events_ids[keys] for keys in motor_event_ids]
    except KeyError as e:
        print(f"Erreur: Événement manquant {e}")
        print(f"Événements disponibles: {list(events_ids.keys())}")
        return None, None
    
    # Mappage des événements
    event_id = {
        'left_hand': motor_event_ids_effectives[0],
        'right_hand': motor_event_ids_effectives[1],
        'feet': motor_event_ids_effectives[2],
        'tongue': motor_event_ids_effectives[3]
    }
    
    print(f"Événements mappés: {event_id}")
    
    # === FILTRAGE AMÉLIORÉ ===
    raw_filtered = raw.copy()
    
    # 1. Filtrage haute-fréquence pour enlever les artefacts musculaires
    raw_filtered.filter(l_freq=1, h_freq=40, method='iir')
    
    # 2. Filtrage notch pour enlever le 50Hz (secteur européen)
    raw_filtered.notch_filter(freqs=50, method='iir')
    
    # 3. Référence moyenne commune améliorée
    raw_filtered.set_eeg_reference('average')
    
    # === EXTRACTION D'ÉPOQUES ROBUSTE ===
    # Période d'imagerie motrice : 2.5s après le cue jusqu'à 6.5s
    tmin = 2.5  # Après le cue
    tmax = 6.5  # Fin de l'imagerie
    
    # Créer les époques avec rejet automatique d'artefacts
    epochs = mne.Epochs(
        raw_filtered, events, event_id, 
        tmin, tmax, picks=['eeg'], 
        baseline=(None),  # Baseline correction
        preload=True,
        reject=dict(eeg=100e-6),  # Rejet automatique des artefacts > 100µV
        reject_by_annotation=True
    )
    
    print(f"Époques extraites: {len(epochs)} sur {len(events)} événements")
    
    # Vérifier la distribution des classes
    events_in_epochs = epochs.events[:, 2]
    for class_name, event_code in event_id.items():
        count = np.sum(events_in_epochs == event_code)
        print(f"{class_name}: {count} essais")
    
    # Obtenir les données et labels
    X = epochs.get_data()  # (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]
    
    # Mapper vers des indices de classe
    class_mapping = {
        event_id['left_hand']: 0,
        event_id['right_hand']: 1,
        event_id['feet']: 2,
        event_id['tongue']: 3
    }
    
    y = np.array([class_mapping[code] for code in y])
    
    # === TRANSFORMATION TEMPS-FRÉQUENCE AMÉLIORÉE ===
    X_transformed = transform_to_timefreq_3d_improved(X, fs=fs)
    
    return X_transformed, y

def transform_to_timefreq_3d_improved(X, fs=250, n_freqs=20, n_times=25):
    """
    Transformation temps-fréquence améliorée avec focus sur les bandes discriminantes
    
    Améliorations:
    - Focus sur les bandes Alpha (8-13Hz) et Beta (13-30Hz)
    - Utilisation de la transformée de Hilbert pour l'enveloppe
    - Normalisation par canal
    - Downsampling temporel intelligent
    """
    n_trials, n_channels, n_samples = X.shape
    print(f"Transformation TF améliorée: {X.shape}")
    
    # Définir les fréquences d'intérêt (focus sur Alpha et Beta)
    freqs = np.logspace(np.log10(8), np.log10(30), n_freqs)
    
    # Définir les instants temporels
    time_points = np.linspace(0, n_samples-1, n_times).astype(int)
    
    # Initialiser le tenseur de sortie
    X_tf = np.zeros((n_trials, n_freqs, n_times, n_channels))
    
    print("Calcul des représentations temps-fréquence...")
    
    for trial in range(n_trials):
        if trial % 25 == 0:
            print(f"  Essai {trial+1}/{n_trials}")
            
        for ch in range(n_channels):
            signal_data = X[trial, ch, :]
            
            # Calculer la représentation temps-fréquence
            tf_matrix = compute_hilbert_envelope_spectrogram(
                signal_data, freqs, time_points, fs
            )
            
            X_tf[trial, :, :, ch] = tf_matrix
    
    # Normalisation par canal (robuste aux outliers)
    print("Normalisation des données...")
    scaler = RobustScaler()
    
    for ch in range(n_channels):
        # Reshape pour la normalisation
        channel_data = X_tf[:, :, :, ch].reshape(-1, n_freqs * n_times)
        channel_data_scaled = scaler.fit_transform(channel_data)
        X_tf[:, :, :, ch] = channel_data_scaled.reshape(-1, n_freqs, n_times)
    
    # Ajouter la dimension des features
    X_transformed = X_tf.reshape(n_trials, n_freqs, n_times, n_channels, 1)
    
    print(f"Transformation terminée: {X_transformed.shape}")
    return X_transformed

def compute_hilbert_envelope_spectrogram(signal_data, freqs, time_points, fs):
    """
    Calcul de spectrogramme basé sur l'enveloppe de Hilbert
    Plus robuste que la méthode précédente
    """
    n_freqs = len(freqs)
    n_times = len(time_points)
    spectrogram = np.zeros((n_freqs, n_times))
    
    for f_idx, freq in enumerate(freqs):
        # Créer un filtre passe-bande avec une largeur adaptative
        if freq < 15:  # Alpha band
            bandwidth = 2.0
        else:  # Beta band
            bandwidth = 3.0
            
        low_freq = max(freq - bandwidth/2, 0.5)
        high_freq = min(freq + bandwidth/2, fs/2 - 0.5)
        
        # Filtrage passe-bande
        sos = signal.butter(4, [low_freq, high_freq], btype='band', fs=fs, output='sos')
        filtered_signal = signal.sosfilt(sos, signal_data)
        
        # Enveloppe de Hilbert
        analytic_signal = hilbert(filtered_signal)
        envelope = np.abs(analytic_signal)
        
        # Échantillonner aux points temporels d'intérêt
        spectrogram[f_idx, :] = envelope[time_points]
    
    return spectrogram

def apply_spatial_csp(X, y, n_components=6):
    """
    Application du Common Spatial Pattern (CSP) pour améliorer la séparabilité
    """
    print(f"Application du CSP avec {n_components} composantes...")
    
    n_trials, n_freqs, n_times, n_channels, _ = X.shape
    
    # Reformater pour CSP: (n_trials, n_channels, n_features)
    X_csp_input = X.reshape(n_trials, n_channels, n_freqs * n_times)
    
    # Appliquer CSP
    csp = CSP(n_components=n_components, reg='ledoit_wolf')
    X_csp = csp.fit_transform(X_csp_input, y)
    
    # Reformater pour le modèle 3D-CLMI
    # X_csp shape: (n_trials, n_components)
    # On veut: (n_trials, n_freqs, n_times, n_components, 1)
    
    # Répliquer les composantes CSP sur la grille temps-fréquence
    X_csp_3d = np.zeros((n_trials, n_freqs, n_times, n_components, 1))
    
    for trial in range(n_trials):
        for comp in range(n_components):
            # Utiliser la même valeur CSP pour tous les points TF
            X_csp_3d[trial, :, :, comp, 0] = X_csp[trial, comp]
    
    print(f"CSP appliqué: {X_csp_3d.shape}")
    return X_csp_3d

def validate_data_quality(X, y):
    """
    Validation de la qualité des données après preprocessing
    """
    print("\n=== VALIDATION DE LA QUALITÉ ===")
    
    # 1. Vérifier les NaN et Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("❌ ERREUR: Données contiennent des NaN ou Inf!")
        return False
    
    # 2. Vérifier la distribution des classes
    unique_classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {unique_classes}")
    print(f"Comptages: {counts}")
    
    if len(unique_classes) != 4:
        print("❌ ERREUR: Pas exactement 4 classes!")
        return False
    
    # 3. Vérifier la variance des données
    variance_per_feature = np.var(X, axis=0)
    zero_variance_features = np.sum(variance_per_feature == 0)
    
    print(f"Features avec variance nulle: {zero_variance_features}")
    print(f"Variance moyenne: {variance_per_feature.mean():.6f}")
    print(f"Variance std: {variance_per_feature.std():.6f}")
    
    # 4. Vérifier la séparabilité inter-classes
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    X_flat = X.reshape(X.shape[0], -1)
    
    # Test rapide de séparabilité avec LDA
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_flat, y)
        lda_score = lda.score(X_flat, y)
        print(f"Score LDA (séparabilité): {lda_score:.3f}")
        
        if lda_score > 0.3:
            print("✅ Séparabilité acceptable")
        else:
            print("⚠️ Séparabilité faible - vérifier les features")
            
    except Exception as e:
        print(f"⚠️ Impossible de calculer la séparabilité: {e}")
        return False   
    return True

def main_preprocessing():
    """
    Script principal pour le préprocessing des données BCI IV 2a
    """
    print(f"Version {np.__version__}")
    # Définir le chemin vers les fichiers GDF
    GDF_FILES_PATH = "/home/cytech/dev/EEG Classification/data/raw"
    raw_gdfs = []

    # Lire tous les fichiers GDF
    gdf_files = [
        join(GDF_FILES_PATH, f)
        for f in listdir(GDF_FILES_PATH)
        if isfile(join(GDF_FILES_PATH, f)) and f.endswith(".gdf")
    ]

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
            X, y = preprocess_eeg_3dclmi_improved(raw, fs=250)
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

    print(
        f"Forme finale du jeu de données: {X_all.shape}, Forme des labels: {y_all.shape}"
    )
    print(f"Distribution des classes: {np.bincount(y_all)}")

    # Validation de la qualité des données
    if validate_data_quality(X_all, y_all):
        print("✅ Données valides et prêtes pour l'entraînement!")

        # Diviser en ensembles d'entraînement et de validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
        )

        # Sauvegarder les données préprocessées
        print("Sauvegarde des données préprocessées...")
        NPY_FILES_PATH = "/home/cytech/dev/EEG Classification/data/preprocessed"
        if not os.path.exists(NPY_FILES_PATH):
            os.makedirs(NPY_FILES_PATH)

        # Enregistrer les données prétraitées
        np.save(NPY_FILES_PATH + "/X_train_3DCLMI.npy", X_train)
        np.save(NPY_FILES_PATH + "/y_train_3DCLMI.npy", y_train)
        np.save(NPY_FILES_PATH + "/X_val_3DCLMI.npy", X_val)
        np.save(NPY_FILES_PATH + "/y_val_3DCLMI.npy", y_val)

        print("Préprocessing terminé et données sauvegardées!")

    else:
        print("❌ Données invalides - vérifier les erreurs ci-dessus.")
    
if __name__ == "__main__":
    main_preprocessing()
