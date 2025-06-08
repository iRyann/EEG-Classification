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

def validate_data(x, y):
    stats = diagnostic_data_quality(x,y)
    
    if 