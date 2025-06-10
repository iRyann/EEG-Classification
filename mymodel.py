# simple_model.py - Modèle efficace pour BCI sans GPU
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import time

class BCIClassifier:
    """
    Classifieur optimisé pour BCI qui s'entraîne rapidement sans GPU
    """
    
    def __init__(self, model_type='rf'):
        """
        Args:
            model_type: 'rf' (RandomForest), 'svm' (SVM), 'lr' (LogisticRegression)
        """
        self.model_type = model_type
        
        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1  # Parallélisation
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                random_state=42,
                probability=True
            )
        elif model_type == 'lr':
            self.model = LogisticRegression(
                C=1.0,
                solver='liblinear',
                multi_class='ovr',
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")
    
    def train(self, X, y):
        """Entraînement avec validation croisée"""
        print(f"🚀 Entraînement {self.model_type.upper()}...")
        print(f"📊 Données: {X.shape}, Classes: {np.unique(y, return_counts=True)}")
        
        start_time = time.time()
        
        # Validation croisée stratifiée
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"📈 CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Entraînement final
        self.model.fit(X, y)
        
        training_time = time.time() - start_time
        print(f"⏱️  Temps d'entraînement: {training_time:.1f}s")
        
        return cv_scores
    
    def evaluate(self, X_test, y_test):
        """Évaluation détaillée"""
        y_pred = self.model.predict(X_test)
        
        print("\n📊 RAPPORT DE CLASSIFICATION:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Left Hand', 'Right Hand', 'Feet', 'Tongue']))
        
        print("\n🎯 MATRICE DE CONFUSION:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        accuracy = (y_pred == y_test).mean()
        return accuracy, y_pred
    
    def predict_proba(self, X):
        """Probabilités de prédiction"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            return None
    
    def save(self, filepath):
        """Sauvegarder le modèle"""
        joblib.dump(self.model, filepath)
        print(f"💾 Modèle sauvé: {filepath}")
    
    def load(self, filepath):
        """Charger le modèle"""
        self.model = joblib.load(filepath)
        print(f"📂 Modèle chargé: {filepath}")

def compare_models(X, y, test_size=0.2):
    """
    Comparaison rapide de différents modèles
    """
    from sklearn.model_selection import train_test_split
    
    print("🔄 COMPARAISON DE MODÈLES")
    print("=" * 50)
    
    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    models = ['rf', 'svm', 'lr']
    results = {}
    
    for model_type in models:
        print(f"\n🤖 Test {model_type.upper()}:")
        print("-" * 30)
        
        classifier = BCIClassifier(model_type)
        cv_scores = classifier.train(X_train, y_train)
        accuracy, _ = classifier.evaluate(X_test, y_test)
        
        results[model_type] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': accuracy
        }
    
    # Résumé
    print("\n📋 RÉSUMÉ DES PERFORMANCES:")
    print("=" * 50)
    for model_type, metrics in results.items():
        print(f"{model_type.upper():3s} | CV: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f} | Test: {metrics['test_accuracy']:.3f}")
    
    return results

def train_best_model(X, y):
    """
    Entraîner le meilleur modèle basé sur les résultats de comparaison
    """
    print("🏆 ENTRAÎNEMENT DU MODÈLE FINAL")
    print("=" * 40)
    
    # Random Forest généralement le meilleur compromis vitesse/performance
    best_classifier = BCIClassifier('rf')
    
    start_time = time.time()
    cv_scores = best_classifier.train(X, y)
    total_time = time.time() - start_time
    
    print(f"\n✅ Modèle final entraîné en {total_time:.1f}s")
    print(f"📊 Performance finale: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return best_classifier

# ============= FONCTION PRINCIPALE =============

def main():
    """Test du pipeline complet"""
    
    # Charger les données prétraitées
    data_path = 'data/preprocessed/preprocessed_data_csp_simple.npz'
    
    try:
        data = np.load(data_path)
        X = data['data']
        y = data['labels']
        print(f"✅ Données chargées: {X.shape}")
    except FileNotFoundError:
        print(f"❌ Fichier non trouvé: {data_path}")
        print("🔧 Exécutez d'abord le preprocessing corrigé!")
        return
    
    # Comparaison des modèles
    results = compare_models(X, y)
    
    # Entraînement du modèle final
    final_model = train_best_model(X, y)
    
    # Sauvegarde
    final_model.save('data/preprocessed/bci_classifier.pkl')
    
    return final_model, results

if __name__ == "__main__":
    model, results = main()