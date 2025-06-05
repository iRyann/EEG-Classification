import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, Dense, Dropout, Flatten, LSTM, Reshape, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def create_3dclmi_model(input_shape=(30, 30, 22, 1), n_classes=4):
    """
    Create the 3D-CLMI model architecture(2066, 30, 30, 22, 1)
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (time_grid1, time_grid2, channels, features)
    n_classes : int
        Number of classes for classification
        
    Returns:
    --------
    model : tf.keras.Model
        Compiled 3D-CLMI model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # CNN Branch
    # First 3D convolution block
    conv1 = Conv3D(32, kernel_size=(3, 3, 1), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    
    # Second 3D convolution block
    conv2 = Conv3D(64, kernel_size=(3, 3, 1), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)
    
    # Third 3D convolution block
    conv3 = Conv3D(128, kernel_size=(3, 3, 1), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    
    # Flatten CNN output
    cnn_output = Flatten()(pool3)
    
    # LSTM Branch
    # Reshape input for LSTM processing: (batch, time_steps, features)
    # Assuming we want to treat the 30x30 grid as a sequence of 30 time steps with 30*22 features
    lstm_input = Reshape((30, 30*22))(inputs)
    
    # LSTM layers
    lstm1 = LSTM(128, return_sequences=True)(lstm_input)
    lstm2 = LSTM(64)(lstm1)
    
    # Merge CNN and LSTM branches
    merged = Concatenate()([cnn_output, lstm2])
    
    # Fully connected layers
    dense1 = Dense(256, activation='relu')(merged)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(128, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    
    # Output layer
    outputs = Dense(n_classes, activation='softmax')(dropout2)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100):
    """
    Train the 3D-CLMI model
    
    Parameters:
    -----------
    model : tf.keras.Model
        The 3D-CLMI model
    X_train, y_train : numpy.ndarray
        Training data and labels
    X_val, y_val : numpy.ndarray
        Validation data and labels
    batch_size : int
        Batch size for training
    epochs : int
        Maximum number of epochs
        
    Returns:
    --------
    history : tf.keras.callbacks.History
        Training history
    """
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        ModelCheckpoint('best_3dclmi_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    Parameters:
    -----------
    model : tf.keras.Model
        Trained 3D-CLMI model
    X_test, y_test : numpy.ndarray
        Test data and labels
        
    Returns:
    --------
    metrics : dict
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    class_report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Left Hand', 'Right Hand', 'Feet', 'Tongue'],
                yticklabels=['Left Hand', 'Right Hand', 'Feet', 'Tongue'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def plot_training_history(history):
    """
    Plot training and validation metrics
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History
        Training history
    """
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')

# Load the preprocessed data
def load_and_train():
    """
    Load preprocessed data and train the 3D-CLMI model
    """
    try:
        X_train = np.load('X_train_3DCLMI.npy')
        y_train = np.load('y_train_3DCLMI.npy')
        X_val = np.load('X_val_3DCLMI.npy')
        y_val = np.load('y_val_3DCLMI.npy')
        
        print(f"Loaded training data: {X_train.shape}, {y_train.shape}")
        print(f"Loaded validation data: {X_val.shape}, {y_val.shape}")
        
        # Create and train model
        input_shape = X_train.shape[1:]
        model = create_3dclmi_model(input_shape=input_shape)
        print(model.summary())
        
        # Train the model
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate on validation data
        metrics = evaluate_model(model, X_val, y_val)
        print(f"Validation accuracy: {metrics['accuracy']:.4f}")
        print("Classification report:")
        for cls, values in metrics['classification_report'].items():
            if isinstance(values, dict):
                print(f"  Class {cls}: Precision={values['precision']:.3f}, Recall={values['recall']:.3f}, F1={values['f1-score']:.3f}")
        
        # Save the final model
        model.save('final_3dclmi_model.h5')
        print("Model saved as 'final_3dclmi_model.h5'")
        
    except Exception as e:
        print(f"Error in loading data or training: {e}")

if __name__ == "__main__":
    load_and_train()