import numpy as np
import matplotlib.pyplot as plt

"""
Data utils
"""

# ------ Data augmentation ------

def add_gaussian_noise(x):
    x_noised =  x.copy()
    x_std = np.std(x) * 0.05
    noise = np.random.normal(0, x_std, x.shape)
    x_noised += noise
    return x_noised

def temporal_shift(x):
    time_shift = np.random.randint(-2, 3)
    if time_shift != 0:
        return np.roll(x, shift=time_shift, axis=2)
    return x
    
def rescale(x):
        scale_factor = np.random.uniform(0.9, 1.1)
        return x * scale_factor

def augment_eeg_data(x, y, augmentation_factor=2):
    print("=== AUGMENTATION DE DONNÉES ===")
    x_res = [x]
    y_res = [y]

    for _ in range(augmentation_factor):
        x_temp = add_gaussian_noise(x)
        x_temp = temporal_shift(x_temp)
        x_temp = rescale(x_temp)
        x_res.append(x_temp)
        y_res.append(y.copy())

    x_res = np.concatenate(x_res, axis=0)
    y_res = np.concatenate(y_res, axis=0)

    return x_res, y_res

# ------ Data validation ------

#TODO: Possibilité de retourne le dictionnaire de stats...
def diagnostic_data_quality(x, y, mt= 'cnn_2d', class_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue'], verbose = False):
    """
    Diagnostic complet de la qualité des données
    """
    PRINTV = print if verbose else lambda *args, **kwargs: None 

    # 1. Distribution des classes
    class_counts = np.bincount(y)
    class_distribution = class_counts / len(y)

    # 2. Statistiques des features
    data_shape = x.shape
    data_min = x.min()
    data_max = x.max()
    data_mean = x.mean()
    data_std = x.std()

    # Stockage dans un dictionnaire
    stats = {
        "class_counts": class_counts.tolist(),
        "class_distribution": class_distribution.tolist(),
        "data_shape": data_shape,
        "data_min": float(data_min),
        "data_max": float(data_max),
        "data_mean": float(data_mean),
        "data_std": float(data_std)
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for class_idx in range(4):
        class_mask = y == class_idx
        if np.sum(class_mask) > 0:
            # Adapter le calcul de la moyenne selon le type de modèle
            # Déterminer les axes à moyenner selon le format attendu
            if mt == 'cnn_2d':
                # Format attendu: (n_epochs, n_channels, n_freqs, n_times)
                # On moyenne sur epochs et channels
                class_data = x[class_mask].mean(axis=(0, 1))
            elif mt == 'lstm':
                # Format attendu: (n_epochs, n_times, n_channels, n_freqs)
                # On moyenne sur epochs et channels
                class_data = x[class_mask].mean(axis=(0, 2))
            else:
                # Par défaut, on suppose (n_epochs, n_channels, n_freqs, n_times)
                class_data = x[class_mask].mean(axis=(0, 1))

            im = axes[class_idx].imshow(class_data, aspect='auto', origin='lower',
                                        cmap='viridis', extent=[0, class_data.shape[-1], 8, 30])
            axes[class_idx].set_title(f'{class_names[class_idx]} (n={np.sum(class_mask)})')
            axes[class_idx].set_xlabel('Temps (bins)')
            axes[class_idx].set_ylabel('Fréquence (Hz)')
            plt.colorbar(im, ax=axes[class_idx])
    
    plt.tight_layout()
    plt.savefig('spectrograms_by_class.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    PRINTV("\n=== ANALYSE DE VARIANCE INTER-CLASSES ===")
    
    x_flattened = x.reshape(x.shape[0], -1)  # Aplatir toutes les dimensions
    
    total_variance = np.var(x_flattened, axis=0)
    within_class_variances = []
    
    for class_idx in range(4):
        class_mask = y == class_idx
        if np.sum(class_mask) > 1:
            class_var = np.var(x_flattened[class_mask], axis=0)
            within_class_variances.append(class_var)
    
    within_class_variance = np.mean(within_class_variances, axis=0)
    between_class_variance = total_variance - within_class_variance
    
    # Ratio de Fisher (between/within variance)
    fisher_ratio = between_class_variance / (within_class_variance + 1e-8)
    
    stats.update({
        "total_variance_mean": float(total_variance.mean()),
        "within_class_variance_mean": float(within_class_variance.mean()),
        "between_class_variance_mean": float(between_class_variance.mean()),
        "fisher_ratio_mean": float(fisher_ratio.mean()),
        "fisher_ratio_above_1_pct": float((fisher_ratio > 1).mean() * 100)
    })

    # 4. Vérifier la séparabilité inter-classes
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    x_flat = x.reshape(x.shape[0], -1)
    
    # Test rapide de séparabilité avec LDA
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_flat, y)
        lda_score = lda.score(x_flat, y)
        PRINTV(f"Score LDA (séparabilité): {lda_score:.3f}")
        
        if lda_score > 0.3:
            PRINTV("✅ Séparabilité acceptable")
        else:
            PRINTV("⚠️ Séparabilité faible - vérifier les features")
            
    except Exception as e:
        PRINTV(f"⚠️ Impossible de calculer la séparabilité: {e}")

    # Acceptabilité :
    
    print(stats)
    decision = True if input("Accepter (T) ou Refuser (F)") == "T" else False
    
    return decision