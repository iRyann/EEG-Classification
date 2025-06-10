# keras_bci_model.py - Modèle Keras optimisé pour BCI IV 2a
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import time
import os

# Configuration pour reproductibilité
tf.random.set_seed(42)
np.random.seed(42)

class BCIKerasModel:
    """
    Modèle Keras optimisé pour classification BCI d'imagerie motrice
    """
    
    def __init__(self, input_shape, num_classes=4, model_type='deep_csp'):
        """
        Args:
            input_shape: Shape des données d'entrée (ex: (6,) pour CSP)
            num_classes: Nombre de classes (4 pour BCI IV 2a)
            model_type: 'deep_csp', 'cnn_1d', 'lstm', 'attention'
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_type = model_type
        self.model = None
        self.history = None
        
        # Construire le modèle
        self._build_model()
    
    def _build_model(self):
        """Construction du modèle selon le type choisi"""
        
        if self.model_type == 'deep_csp':
            self.model = self._build_deep_csp_model()
        elif self.model_type == 'cnn_1d':
            self.model = self._build_cnn_1d_model()
        elif self.model_type == 'lstm':
            self.model = self._build_lstm_model()
        elif self.model_type == 'attention':
            self.model = self._build_attention_model()
        else:
            raise ValueError(f"Type de modèle non supporté: {self.model_type}")
    
    def _build_deep_csp_model(self):
        """
        Modèle dense profond pour features CSP
        Optimisé pour les 6 composantes CSP
        """
        model = models.Sequential([
            # Couches denses avec BatchNorm et Dropout
            layers.Dense(128, activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            # Couche de sortie
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_cnn_1d_model(self):
        """
        CNN 1D pour données temporelles EEG
        Nécessite des données avec dimension temporelle
        """
        model = models.Sequential([
            # Première couche CNN
            layers.Conv1D(32, kernel_size=3, activation='relu', 
                         input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Deuxième couche CNN
            layers.Conv1D(64, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # Troisième couche CNN
            layers.Conv1D(128, kernel_size=3, activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Couches denses finales
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_lstm_model(self):
        """
        LSTM pour capturer les dépendances temporelles
        """
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_attention_model(self):
        """
        Modèle avec mécanisme d'attention
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Couches denses avec attention
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Mécanisme d'attention simple
        attention = layers.Dense(128, activation='tanh')(x)
        attention = layers.Dense(1, activation='sigmoid')(attention)
        x = layers.Multiply()([x, attention])
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32, 
              early_stopping=True, verbose=1):
        """
        Entraînement du modèle avec callbacks
        """
        print(f"🚀 Entraînement modèle Keras {self.model_type.upper()}...")
        print(f"📊 Données: {X.shape}, Classes: {np.unique(y, return_counts=True)}")
        
        # Conversion en categorical
        y_cat = to_categorical(y, num_classes=self.num_classes)
        
        # Callbacks
        callback_list = [
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.7, patience=10, 
                min_lr=1e-6, verbose=verbose
            )
        ]
        
        if early_stopping:
            callback_list.append(
                callbacks.EarlyStopping(
                    monitor='val_loss', patience=20, 
                    restore_best_weights=True, verbose=verbose
                )
            )
        
        # Entraînement
        start_time = time.time()
        
        self.history = self.model.fit(
            X, y_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=verbose
        )
        
        training_time = time.time() - start_time
        print(f"⏱️  Temps d'entraînement: {training_time:.1f}s")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Évaluation détaillée"""
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Prédictions
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métriques
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        print(f"\n📊 PERFORMANCE TEST:")
        print(f"🎯 Accuracy: {test_accuracy:.3f}")
        print(f"📉 Loss: {test_loss:.3f}")
        
        print("\n📊 RAPPORT DE CLASSIFICATION:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue']))
        
        print("\n🎯 MATRICE DE CONFUSION:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return test_accuracy, y_pred, y_pred_proba
    
    def plot_training_history(self, save_path=None):
        """Graphique de l'historique d'entraînement"""
        if self.history is None:
            print("❌ Aucun historique d'entraînement disponible")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Graphique sauvé: {save_path}")
        
        plt.show()
    
    def save_model(self, filepath):
        """Sauvegarder le modèle"""
        self.model.save(filepath)
        print(f"💾 Modèle Keras sauvé: {filepath}")
    
    def load_model(self, filepath):
        """Charger le modèle"""
        self.model = keras.models.load_model(filepath)
        print(f"📂 Modèle Keras chargé: {filepath}")

def cross_validate_keras(X, y, model_type='deep_csp', n_splits=5):
    """
    Validation croisée pour modèle Keras
    """
    print(f"🔄 VALIDATION CROISÉE KERAS - {model_type.upper()}")
    print("=" * 50)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n📁 Fold {fold + 1}/{n_splits}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Créer et entraîner le modèle
        model = BCIKerasModel(
            input_shape=(X.shape[1],),
            num_classes=len(np.unique(y)),
            model_type=model_type
        )
        
        # Entraînement avec moins d'epochs pour CV
        model.train(X_train, y_train, validation_split=0.0, 
                   epochs=50, verbose=0)
        
        # Évaluation
        accuracy, _, _ = model.evaluate(X_val, y_val)
        cv_scores.append(accuracy)
        print(f"  Accuracy: {accuracy:.3f}")
    
    cv_scores = np.array(cv_scores)
    print(f"\n📊 CV Results: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return cv_scores

def compare_keras_models(X, y):
    """
    Comparaison de différents modèles Keras
    """
    print("🔄 COMPARAISON MODÈLES KERAS")
    print("=" * 50)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_to_test = ['deep_csp']  # Commencer par le plus simple
    
    results = {}
    
    for model_type in models_to_test:
        print(f"\n🤖 Test {model_type.upper()}:")
        print("-" * 30)
        
        # Créer et entraîner le modèle
        model = BCIKerasModel(
            input_shape=(X_train.shape[1],),
            num_classes=len(np.unique(y)),
            model_type=model_type
        )
        
        # Entraînement
        history = model.train(X_train, y_train, epochs=100, verbose=1)
        
        # Évaluation
        test_accuracy, _, _ = model.evaluate(X_test, y_test)
        
        # Stocker les résultats
        results[model_type] = {
            'model': model,
            'test_accuracy': test_accuracy,
            'history': history
        }
        
        # Graphique
        model.plot_training_history()
    
    return results

# ============= FONCTION PRINCIPALE =============

def main():
    """Pipeline complet avec Keras"""
    
    # Charger les données prétraitées
    data_path = 'data/preprocessed/preprocessed_data_csp_simple.npz'
    
    try:
        data = np.load(data_path)
        X = data['data']
        y = data['labels']
        print(f"✅ Données chargées: {X.shape}")
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {data_path}")
        print("🔧 Exécutez d'abord le preprocessing!")
        return
    
    # Comparaison des modèles Keras
    results = compare_keras_models(X, y)
    
    # Sauvegarder le meilleur modèle
    best_model_type = max(results.keys(), 
                         key=lambda k: results[k]['test_accuracy'])
    best_model = results[best_model_type]['model']
    
    # Sauvegarder
    os.makedirs('data/preprocessed', exist_ok=True)
    best_model.save_model('data/preprocessed/bci_keras_model.h5')
    
    print(f"\n🏆 Meilleur modèle: {best_model_type}")
    print(f"📊 Performance: {results[best_model_type]['test_accuracy']:.3f}")
    
    return results

if __name__ == "__main__":
    results = main()