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
from utils.model import diagnostic_data_quality

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
    diagnostic_data_quality(data['data'], data['labels'], verbose=True)

if __name__ == "__main__":
    main()