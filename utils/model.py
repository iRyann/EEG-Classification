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

def diagnostic_data_quality_fixed(x, y, verbose=True):
    """Version optimisée sans blocage LDA"""
    
    # Réduction de dimensionnalité avant LDA
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
    x_flat = x.reshape(x.shape[0], -1)
    
    # PCA pour réduire dimensionnalité
    n_components = min(500, x_flat.shape[1], x_flat.shape[0]-1)
    pca = PCA(n_components=n_components)
    x_reduced = pca.fit_transform(x_flat)
    
    print(f"Réduction PCA: {x_flat.shape[1]} -> {x_reduced.shape[1]} features")
    print(f"Variance expliquée: {pca.explained_variance_ratio_.sum():.3f}")
    
    # LDA sur données réduites
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(x_reduced, y)
        lda_score = lda.score(x_reduced, y)
        print(f"Score LDA: {lda_score:.3f}")
        
        # Analyse de variance corrigée
        analyze_class_separability(x_reduced, y)
        
    except Exception as e:
        print(f"Erreur LDA: {e}")
    
    return True

def analyze_class_separability(x, y):
    """Analyse détaillée de la séparabilité"""
    from scipy.stats import f_oneway
    
    # Test ANOVA sur chaque feature
    f_stats = []
    p_values = []
    
    for feature_idx in range(min(100, x.shape[1])):  # Limiter pour performance
        groups = [x[y == class_idx, feature_idx] for class_idx in range(4)]
        groups = [g for g in groups if len(g) > 1]  # Filtrer groupes vides
        
        if len(groups) == 4:
            f_stat, p_val = f_oneway(*groups)
            f_stats.append(f_stat)
            p_values.append(p_val)
    
    f_stats = np.array(f_stats)
    p_values = np.array(p_values)
    
    print(f"Features significatives (p<0.05): {(p_values < 0.05).sum()}/{len(p_values)}")
    print(f"F-statistic moyenne: {f_stats.mean():.3f}")
    print(f"F-statistic max: {f_stats.max():.3f}")