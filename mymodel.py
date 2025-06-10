import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, Dropout, Flatten, LSTM, Reshape, Concatenate, BatchNormalization, Bidirectional, GlobalAveragePooling3D, Multiply, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.regularizers import l2
from utils.model import diagnostic_data_quality, analyze_class_separability

DATA_PATH = 'data/preprocessed/preprocessed_data.npz'
PREPROCESSED_DATA = None

def load_data():
    global PREPROCESSED_DATA
    if PREPROCESSED_DATA is None:
        try:
            PREPROCESSED_DATA = np.load(DATA_PATH, allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    return PREPROCESSED_DATA

def main():
    data = load_data()
    if 'data' not in data or 'labels' not in data:
        raise ValueError("Data must contain 'data' and 'labels' keys.")
    x = data['data']
    y = data['labels']

    print("=== DATA LOADED ===")
    print(f"Data shape: {x.shape}, Labels shape: {y.shape}")
    # Data quality diagnostics
    diagnostic_data_quality(x, y, verbose=True)
    # Analyze class separability
    analyze_class_separability(x, y, verbose=True)
    # Visualize data distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.title("Class Distribution")
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.show()
    # Visualize sample data
    sample_index = np.random.randint(0, x.shape[0])
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(x[sample_index, :, :, 0], aspect='auto', cmap='viridis')
    plt.title(f"Sample EEG Data - Class {y[sample_index]}")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.plot(x[sample_index, :, 0, 0], label='Channel 1')
    plt.plot(x[sample_index, :, 1, 0], label='Channel 2')
    plt.title("Sample EEG Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()
    # PCA for dimensionality reduction
    

if __name__ == "__main__":
    main()