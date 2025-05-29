import os
import numpy as np
import mne
from os import listdir
from os.path import isfile, join
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set the path to your GDF files
GDF_FILES_PATH = '/home/cytech/dev/EEG Classification/data'
raw_gdfs = []

# Read all GDF files
for gdf_file in [join(GDF_FILES_PATH, f) for f in listdir(GDF_FILES_PATH) if isfile(join(GDF_FILES_PATH, f))]:
    print(f"Loading {gdf_file}")
    raw_gdfs.append(mne.io.read_raw_gdf(gdf_file, eog=[22, 23, 24], preload=True))

# Function to preprocess each raw GDF file
def preprocess_eeg(raw, event_id=None):
    """
    Preprocess the EEG data according to the 3D-CLMI pipeline
    
    Parameters:
    -----------
    raw : mne.io.Raw
        Raw EEG data
    event_id : dict
        Dictionary mapping event names to event codes
        
    Returns:
    --------
    X : numpy.ndarray
        Preprocessed EEG data of shape (n_trials, 30, 30, 22, 1)
    y : numpy.ndarray
        Labels of shape (n_trials,)
    """
    
    # Extract events
    events, events_ids = mne.events_from_annotations(raw)

    # Codes des événements correspondant aux 4 classes d'imagerie motrice
    motor_event_ids = ['769', '770', '771', '772']
    motor_event_ids_effectives = [events_ids[keys] for keys in motor_event_ids]
    
    # Mappage des événements pour mne.Epochs
    event_id = {
        'left_hand': motor_event_ids_effectives[0],
        'right_hand': motor_event_ids_effectives[1],
        'feet': motor_event_ids_effectives[2],
        'tongue': motor_event_ids_effectives[3]
    }
    
    # Apply bandpass filter (8-30 Hz) to capture Alpha and Beta rhythms
    raw.filter(8, 30, method='iir')
    
    # Apply CAR (Common Average Reference)
    raw.set_eeg_reference('average')
    
    # Extract epochs
    # Start 2.5s after cue (cue is at t=2s, so 4.5s from start)
    # End 4.5s after cue (6.5s from start)
    tmin = 4.5  # 2.5s after cue
    tmax = 6.5  # 4.5s after cue
    
    # Create epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                        picks=['eeg'], baseline=None, preload=True)
    
    # Get data and labels
    X = epochs.get_data()  # Shape: (n_trials, n_channels, n_times)
    y = epochs.events[:, 2]
    
    # Map event codes to 0-indexed classes
    class_mapping = {
        event_id['left_hand']: 0,
        event_id['right_hand']: 1,
        event_id['feet']: 2,
        event_id['tongue']: 3
    }
    
    y = np.array([class_mapping[code] for code in y])
    
    # Transform data into the required shape (None, 30, 30, 22, 1)
    X_transformed = transform_to_model_input(X)
    
    return X_transformed, y

def transform_to_model_input(X):
    """
    Transform the epoched EEG data into the format required by the 3D-CLMI model
    
    Parameters:
    -----------
    X : numpy.ndarray
        Epoched EEG data of shape (n_trials, n_channels, n_times)
        
    Returns:
    --------
    X_transformed : numpy.ndarray
        Transformed data of shape (n_trials, 22, 22, 22, 1)
    """
    n_trials, n_channels, n_times = X.shape
    
    # Since the exact method for transforming the 500 time points to 22x22 grid
    # is not explicitly described, we'll use a reasonable approach:
    # 1. Reshape the 500 time points to approximately 22x22 = 484 points 
    # 2. Use interpolation to get exactly 22x22 grid
    
    # Method 1: Simple reshaping with padding or truncation
    target_length = 22 * 22
    X_reshaped = np.zeros((n_trials, n_channels, target_length))
    
    for i in range(n_trials):
        for j in range(n_channels):
            # If n_times < target_length, pad with zeros
            # If n_times > target_length, truncate
            if n_times <= target_length:
                X_reshaped[i, j, :n_times] = X[i, j, :]
            else:
                # Alternatively, we could use interpolation here
                X_reshaped[i, j, :] = signal.resample(X[i, j, :], target_length)
    
    # Reshape to (n_trials, 22, 22, n_channels)
    X_grid = X_reshaped.reshape(n_trials, n_channels, 22, 22)
    
    # Transpose to get (n_trials, 22, 22, n_channels)
    X_grid = np.transpose(X_grid, (0, 2, 3, 1))
    
    # Add channel dimension for CNN: (n_trials, 22, 22, n_channels, 1)
    X_transformed = X_grid.reshape(n_trials, 22, 22, n_channels, 1)
    
    return X_transformed

# Process each GDF file
processed_data = []
for i, raw in enumerate(raw_gdfs):
    print(f"Processing file {i+1}/{len(raw_gdfs)}")
    try:
        X, y = preprocess_eeg(raw)
        processed_data.append((X, y))
        print(f"  Extracted {len(y)} trials with shape {X.shape}")
    except Exception as e:
        print(f"  Error processing file: {e}")

# Combine all processed data
X_all = np.concatenate([data[0] for data in processed_data], axis=0)
y_all = np.concatenate([data[1] for data in processed_data], axis=0)

print(f"Final dataset shape: {X_all.shape}, Labels shape: {y_all.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# Save preprocessed data
np.save('X_train_3DCLMI.npy', X_train)
np.save('y_train_3DCLMI.npy', y_train)
np.save('X_val_3DCLMI.npy', X_val)
np.save('y_val_3DCLMI.npy', y_val)

print("Preprocessing completed and data saved!")

# Alternative method for time-frequency transformation (if needed)
def transform_to_timefreq_grid(X, fs=250):
    """
    Alternative method using time-frequency decomposition
    This might be closer to what the 3D-CLMI paper actually did
    
    Parameters:
    -----------
    X : numpy.ndarray
        Epoched EEG data of shape (n_trials, n_channels, n_times)
    fs : int
        Sampling frequency
        
    Returns:
    --------
    X_transformed : numpy.ndarray
        Transformed data of shape (n_trials, 30, 30, n_channels, 1)
    """
    n_trials, n_channels, n_times = X.shape
    
    # Define parameters for time-frequency analysis
    freqs = np.linspace(8, 30, 30)  # 30 frequency bins from 8-30 Hz
    n_cycles = freqs / 2.  # Higher frequencies get more cycles
    
    # Create output array
    X_tf = np.zeros((n_trials, n_channels, len(freqs), 30))
    
    for i in range(n_trials):
        for j in range(n_channels):
            # Compute time-frequency representation using Morlet wavelets
            # This is a simplified version - in practice you'd use mne.time_frequency
            signal_data = X[i, j, :]
            times = np.linspace(0, n_times/fs, 30)  # 30 time bins
            
            # Resample signal to match desired time bins
            resampled_signal = signal.resample(signal_data, 30)
            
            # For each frequency, compute power
            for f_idx, freq in enumerate(freqs):
                # Simple approximation of band power at each frequency
                # In practice, you'd use proper wavelet convolution
                band_signal = signal.butter(4, [freq-1, freq+1], 'bandpass', fs=fs)
                filtered = signal.filtfilt(*band_signal, resampled_signal)
                X_tf[i, j, f_idx, :] = np.abs(filtered) ** 2
    
    # Transpose to get (n_trials, freqs, times, channels)
    X_grid = np.transpose(X_tf, (0, 2, 3, 1))
    
    # Add channel dimension for CNN: (n_trials, freqs, times, channels, 1)
    X_transformed = X_grid.reshape(n_trials, 30, 30, n_channels, 1)
    
    return X_transformed