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

def load_and_prepare_raw_eeg_data():
    """
    Charger et préparer les données EEG brutes pour CNN/LSTM
    Alternative aux features CSP pour avoir des données temporelles
    """
    print("📊 CHARGEMENT DONNÉES EEG TEMPORELLES...")
    
    try:
        # Essayer de charger les données preprocessées avec information temporelle
        data_path = 'data/preprocessed/eeg_temporal_data.npz'
        if os.path.exists(data_path):
            data = np.load(data_path)
            return data['data'], data['labels']
        else:
            print("⚠️  Données temporelles non trouvées. Utilisation des features CSP adaptées.")
            return None, None
    except:
        return None, None

def create_temporal_data_from_csp(X_csp, y, time_steps=10):
    """
    Créer des données pseudo-temporelles à partir des features CSP
    Pour tester CNN1D et LSTM même sans données temporelles originales
    """
    print(f"🔄 Création de données temporelles à partir de CSP...")
    
    n_samples, n_features = X_csp.shape
    
    # Créer une séquence temporelle en dupliquant et en ajoutant du bruit
    X_temporal = np.zeros((n_samples, time_steps, n_features))
    
    for i in range(n_samples):
        for t in range(time_steps):
            # Base: les features CSP originales
            base_features = X_csp[i]
            
            # Ajouter variation temporelle simulée
            noise_factor = 0.1 * np.sin(2 * np.pi * t / time_steps)
            temporal_variation = base_features * (1 + noise_factor * np.random.normal(0, 0.1, n_features))
            
            X_temporal[i, t, :] = temporal_variation
    
    print(f"✅ Données temporelles créées: {X_temporal.shape}")
    return X_temporal, y
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

def prepare_data_for_model(X, y, model_type):
    """
    Adapter les données selon le type de modèle
    """
    if model_type == 'deep_csp' or model_type == 'attention':
        # Données CSP directes (shape: n_samples, n_features)
        return X, y
    
    elif model_type == 'cnn_1d' or model_type == 'lstm':
        # Pour CNN1D et LSTM, on a besoin d'une dimension temporelle
        # Simulation: on reshape les features CSP en séquence temporelle
        if len(X.shape) == 2:
            # Transformer (n_samples, n_features) -> (n_samples, time_steps, features)
            n_samples, n_features = X.shape
            time_steps = min(n_features, 6)  # Maximum 6 time steps
            features_per_step = n_features // time_steps
            
            if features_per_step == 0:
                features_per_step = 1
                time_steps = n_features
            
            # Reshape et padding si nécessaire
            X_reshaped = X[:, :time_steps * features_per_step]
            X_reshaped = X_reshaped.reshape(n_samples, time_steps, features_per_step)
            
            print(f"📊 Données adaptées pour {model_type}: {X.shape} -> {X_reshaped.shape}")
            return X_reshaped, y
        else:
            return X, y
    
    return X, y

def compare_keras_models(X, y, test_all_models=True):
    """
    Comparaison de différents modèles Keras
    """
    print("🔄 COMPARAISON MODÈLES KERAS")
    print("=" * 50)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if test_all_models:
        models_to_test = ['deep_csp', 'cnn_1d', 'lstm', 'attention']
    else:
        models_to_test = ['deep_csp']
    
    results = {}
    
    for model_type in models_to_test:
        print(f"\n🤖 Test {model_type.upper()}:")
        print("-" * 30)
        
        try:
            # Adapter les données pour ce type de modèle
            X_train_adapted, y_train_adapted = prepare_data_for_model(X_train, y_train, model_type)
            X_test_adapted, y_test_adapted = prepare_data_for_model(X_test, y_test, model_type)
            
            # Déterminer la shape d'entrée
            if len(X_train_adapted.shape) == 2:
                input_shape = (X_train_adapted.shape[1],)
            else:
                input_shape = X_train_adapted.shape[1:]
            
            print(f"📊 Input shape pour {model_type}: {input_shape}")
            
            # Créer et entraîner le modèle
            model = BCIKerasModel(
                input_shape=input_shape,
                num_classes=len(np.unique(y)),
                model_type=model_type
            )
            
            # Entraînement avec gestion des erreurs
            history = model.train(
                X_train_adapted, y_train_adapted, 
                epochs=80,  # Réduire pour les tests
                batch_size=16,  # Batch plus petit pour stabilité
                verbose=1
            )
            
            # Évaluation
            test_accuracy, _, _ = model.evaluate(X_test_adapted, y_test_adapted)
            
            # Stocker les résultats
            results[model_type] = {
                'model': model,
                'test_accuracy': test_accuracy,
                'history': history,
                'input_shape': input_shape
            }
            
            print(f"✅ {model_type} terminé - Accuracy: {test_accuracy:.3f}")
            
        except Exception as e:
            print(f"❌ Erreur avec {model_type}: {str(e)}")
            results[model_type] = {
                'error': str(e),
                'test_accuracy': 0.0
            }
    
    # Résumé des résultats
    print("\n📋 RÉSUMÉ DES PERFORMANCES:")
    print("=" * 50)
    for model_type, result in results.items():
        if 'error' in result:
            print(f"{model_type.upper():12s} | ❌ ERREUR: {result['error'][:50]}...")
        else:
            print(f"{model_type.upper():12s} | ✅ Accuracy: {result['test_accuracy']:.3f}")
    
    return results

# ============= FONCTION PRINCIPALE =============

def main(test_all_models=True):
    """Pipeline complet avec Keras - Test de tous les modèles"""
    
    print("🚀 PIPELINE KERAS BCI - TEST COMPLET")
    print("=" * 50)
    
    # Charger les données prétraitées
    data_path = 'data/preprocessed/preprocessed_data_csp_simple.npz'
    
    try:
        data = np.load(data_path)
        X = data['data']
        y = data['labels']
        print(f"✅ Données CSP chargées: {X.shape}")
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {data_path}")
        print("🔧 Exécutez d'abord le preprocessing!")
        return
    
    # Option: créer des données temporelles pour CNN/LSTM
    if test_all_models:
        print("\n📊 Création de données temporelles pour CNN/LSTM...")
        X_temporal, y_temporal = create_temporal_data_from_csp(X, y)
    
    # Comparaison des modèles Keras
    results = compare_keras_models(X, y, test_all_models=test_all_models)
    
    # Afficher les graphiques pour les modèles qui ont réussi
    print("\n📊 GÉNÉRATION DES GRAPHIQUES...")
    for model_type, result in results.items():
        if 'model' in result and 'error' not in result:
            print(f"📈 Graphique pour {model_type}...")
            try:
                result['model'].plot_training_history()
            except Exception as e:
                print(f"⚠️  Erreur graphique {model_type}: {e}")
    
    # Sauvegarder le meilleur modèle
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if valid_results:
        best_model_type = max(valid_results.keys(), 
                             key=lambda k: valid_results[k]['test_accuracy'])
        best_model = valid_results[best_model_type]['model']
        
        # Sauvegarder
        os.makedirs('data/preprocessed', exist_ok=True)
        model_path = f'data/preprocessed/bci_keras_{best_model_type}.h5'
        best_model.save_model(model_path)
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model_type.upper()}")
        print(f"📊 Performance: {valid_results[best_model_type]['test_accuracy']:.3f}")
        print(f"💾 Sauvé: {model_path}")
    else:
        print("❌ Aucun modèle n'a réussi l'entraînement")
    
    return results

def quick_test_single_model(model_type='deep_csp'):
    """Test rapide d'un seul modèle"""
    print(f"⚡ TEST RAPIDE: {model_type.upper()}")
    
    # Charger données
    try:
        data = np.load('data/preprocessed/preprocessed_data_csp_simple.npz')
        X, y = data['data'], data['labels']
    except FileNotFoundError:
        print("❌ Données non trouvées")
        return
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Adapter données
    X_train_adapted, _ = prepare_data_for_model(X_train, y_train, model_type)
    X_test_adapted, _ = prepare_data_for_model(X_test, y_test, model_type)
    
    # Input shape
    if len(X_train_adapted.shape) == 2:
        input_shape = (X_train_adapted.shape[1],)
    else:
        input_shape = X_train_adapted.shape[1:]
    
    # Modèle
    model = BCIKerasModel(input_shape=input_shape, model_type=model_type)
    
    # Entraînement rapide
    model.train(X_train_adapted, y_train, epochs=30, verbose=1)
    
    # Test
    accuracy, _, _ = model.evaluate(X_test_adapted, y_test)
    print(f"🎯 Résultat: {accuracy:.3f}")
    
    return model

if __name__ == "__main__":
    print("🚀 MENU DE TEST KERAS BCI")
    print("=" * 30)
    print("1️⃣  Test complet (tous les modèles)")
    print("2️⃣  Test rapide Deep CSP")
    print("3️⃣  Test rapide CNN 1D")
    print("4️⃣  Test rapide LSTM")
    print("5️⃣  Test rapide Attention")
    
    choice = input("\nChoisissez une option (1-5, ou Entrée pour test complet): ").strip()
    
    if choice == '1' or choice == '':
        print("\n🔄 LANCEMENT TEST COMPLET...")
        results = main(test_all_models=True)
    elif choice == '2':
        model = quick_test_single_model('deep_csp')
    elif choice == '3':
        model = quick_test_single_model('cnn_1d')
    elif choice == '4':
        model = quick_test_single_model('lstm')
    elif choice == '5':
        model = quick_test_single_model('attention')
    else:
        print("❌ Option invalide")
        results = main(test_all_models=True)