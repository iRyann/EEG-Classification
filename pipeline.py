import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_eeg  # Import from the preprocessing script
from mymodel import create_3dclmi_model, train_model, evaluate_model, plot_training_history

# This script demonstrates the end-to-end pipeline from raw GDF files to trained model

def run_pipeline(gdf_files_path, output_path='./output', apply_augmentation=True, augmentation_factor=3):
    """
    Run the complete 3D-CLMI pipeline from preprocessing to evaluation
    
    Parameters:
    -----------
    gdf_files_path : str
        Path to directory containing GDF files
    output_path : str
        Path to directory for output files
    apply_augmentation : bool
        Whether to apply data augmentation
    augmentation_factor : int
        Factor for data augmentation (1=no augmentation, 2=double data, etc.)
    """
    import mne
    from os import listdir
    from os.path import isfile, join
    from sklearn.model_selection import train_test_split
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print("Step 1: Loading and preprocessing GDF files")
    raw_gdfs = []
    
    # Read all GDF files
    for gdf_file in [join(gdf_files_path, f) for f in listdir(gdf_files_path) if isfile(join(gdf_files_path, f))]:
        print(f"  Loading {gdf_file}")
        raw_gdfs.append(mne.io.read_raw_gdf(gdf_file, eog=[22, 23, 24], preload=True))
    
    # Process each GDF file
    processed_data = []
    for i, raw in enumerate(raw_gdfs):
        print(f"  Processing file {i+1}/{len(raw_gdfs)}")
        try:
            X, y = preprocess_eeg(raw, apply_augmentation=apply_augmentation, augmentation_factor=augmentation_factor)
            processed_data.append((X, y))
            print(f"    Extracted {len(y)} trials with shape {X.shape}")
        except Exception as e:
            print(f"    Error processing file: {e}")
    
    # Combine all processed data
    X_all = np.concatenate([data[0] for data in processed_data], axis=0)
    y_all = np.concatenate([data[1] for data in processed_data], axis=0)
    
    print(f"  Final dataset shape: {X_all.shape}, Labels shape: {y_all.shape}")
    
    return X_all, y_all

    # # Split into training, validation, and test sets
    # X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # print(f"  Train set: {X_train.shape}, {y_train.shape}")
    # print(f"  Validation set: {X_val.shape}, {y_val.shape}")
    # print(f"  Test set: {X_test.shape}, {y_test.shape}")
    
    # # Save preprocessed data
    # np.save(join(output_path, 'X_train.npy'), X_train)
    # np.save(join(output_path, 'y_train.npy'), y_train)
    # np.save(join(output_path, 'X_val.npy'), X_val)
    # np.save(join(output_path, 'y_val.npy'), y_val)
    # np.save(join(output_path, 'X_test.npy'), X_test)
    # np.save(join(output_path, 'y_test.npy'), y_test)
    
    # print("Step 2: Creating 3D-CLMI model")
    # # Create model
    # input_shape = X_train.shape[1:]
    # model = create_3dclmi_model(input_shape=input_shape)
    # model.summary()
    
    # print("Step 3: Training the model")
    # # Train model
    # history = train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100)
    
    # # Plot training history
    # plot_training_history(history)
    # plt.savefig(join(output_path, 'training_history.png'))
    
    # print("Step 4: Evaluating the model")
    # # Evaluate model
    # metrics = evaluate_model(model, X_test, y_test)
    # print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    
    # # Print classification report
    # print("  Classification report:")
    # for cls, values in metrics['classification_report'].items():
    #     if isinstance(values, dict):
    #         print(f"    Class {cls}: Precision={values['precision']:.3f}, Recall={values['recall']:.3f}, F1={values['f1-score']:.3f}")
    
    # # Save confusion matrix
    # plt.figure(figsize=(10, 8))
    # import seaborn as sns
    # sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
    #             xticklabels=['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
    #             yticklabels=['Left Hand', 'Right Hand', 'Feet', 'Tongue'])
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.title('Confusion Matrix')
    # plt.savefig(join(output_path, 'confusion_matrix.png'))
    
    # # Save the final model
    # model.save(join(output_path, 'final_3dclmi_model.h5'))
    # print(f"Model saved as '{join(output_path, 'final_3dclmi_model.h5')}'")
    
    # print("Pipeline completed successfully!")

if __name__ == "__main__":
    # Example usage
    GDF_FILES_PATH = '/home/cytech/Desktop/EEG Classification/data'
    x,y = run_pipeline(GDF_FILES_PATH)