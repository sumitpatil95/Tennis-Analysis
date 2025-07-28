"""
Classical Machine Learning Models for Tennis Action Recognition
Implements Random Forest, SVM, and other classical ML approaches
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import json


class ClassicalMLTrainer:
    """Train and evaluate classical ML models for tennis action recognition"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.action_names = ["backhand", "forehand", "ready_position", "serve"]
        self.results = {}
        
    def prepare_data(self, X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray) -> Tuple:
        """Prepare and scale data for classical ML models"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_val: np.ndarray, y_val: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest classifier with hyperparameter tuning"""
        print("Training Random Forest...")
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_macro', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_pred = best_rf.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        print(f"Random Forest - Best params: {grid_search.best_params_}")
        print(f"Random Forest - Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        self.models['random_forest'] = best_rf
        self.results['random_forest'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        
        return best_rf
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> SVC:
        """Train SVM classifier with hyperparameter tuning"""
        print("Training SVM...")
        
        # Hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        svm = SVC(random_state=42, probability=True)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_svm = grid_search.best_estimator_
        
        # Evaluate on validation set
        val_pred = best_svm.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        print(f"SVM - Best params: {grid_search.best_params_}")
        print(f"SVM - Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        self.models['svm'] = best_svm
        self.results['svm'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        
        return best_svm
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray) -> LogisticRegression:
        """Train Logistic Regression classifier"""
        print("Training Logistic Regression...")
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga']
        }
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_lr = grid_search.best_estimator_
        
        val_pred = best_lr.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        print(f"Logistic Regression - Best params: {grid_search.best_params_}")
        print(f"Logistic Regression - Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        self.models['logistic_regression'] = best_lr
        self.results['logistic_regression'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        
        return best_lr
    
    def train_knn(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray) -> KNeighborsClassifier:
        """Train K-Nearest Neighbors classifier"""
        print("Training K-Nearest Neighbors...")
        
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
        
        knn = KNeighborsClassifier()
        
        grid_search = GridSearchCV(
            knn, param_grid, cv=5, scoring='f1_macro',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_knn = grid_search.best_estimator_
        
        val_pred = best_knn.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        print(f"KNN - Best params: {grid_search.best_params_}")
        print(f"KNN - Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
        
        self.models['knn'] = best_knn
        self.results['knn'] = {
            'best_params': grid_search.best_params_,
            'val_accuracy': val_accuracy,
            'val_f1': val_f1
        }
        
        return best_knn
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all trained models on test set"""
        test_results = {}
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            
            # Classification report
            class_report = classification_report(
                y_test, y_pred, 
                target_names=self.action_names,
                output_dict=True
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            test_results[model_name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'classification_report': class_report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred.tolist(),
                'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
            
            print(f"{model_name} - Test Accuracy: {accuracy:.4f}, Test F1 (macro): {f1_macro:.4f}")
        
        return test_results
    
    def plot_confusion_matrices(self, test_results: Dict, save_path: str = None):
        """Plot confusion matrices for all models"""
        n_models = len(test_results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (model_name, results) in enumerate(test_results.items()):
            cm = np.array(results['confusion_matrix'])
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=self.action_names,
                yticklabels=self.action_names,
                ax=axes[idx]
            )
            
            axes[idx].set_title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_model_comparison(self, test_results: Dict, save_path: str = None):
        """Plot model performance comparison"""
        models = list(test_results.keys())
        accuracies = [test_results[model]['accuracy'] for model in models]
        f1_scores = [test_results[model]['f1_macro'] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='F1-Score (Macro)', alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([model.replace('_', ' ').title() for model in models])
        ax.legend()
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_models(self, save_dir: str):
        """Save trained models and scalers"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = os.path.join(save_dir, f"scaler_{scaler_name}.pkl")
            joblib.dump(scaler, scaler_path)
            print(f"Saved {scaler_name} scaler to {scaler_path}")
        
        # Save results
        results_path = os.path.join(save_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Saved training results to {results_path}")
    
    def load_models(self, save_dir: str):
        """Load trained models and scalers"""
        import os
        
        # Load models
        for model_name in ['random_forest', 'svm', 'logistic_regression', 'knn']:
            model_path = os.path.join(save_dir, f"{model_name}.pkl")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} from {model_path}")
        
        # Load scalers
        scaler_path = os.path.join(save_dir, "scaler_standard.pkl")
        if os.path.exists(scaler_path):
            self.scalers['standard'] = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")


def main():
    """Example usage of classical ML trainer"""
    from src.data.dataset_processor import TennisDatasetProcessor
    
    # Load processed data
    processor = TennisDatasetProcessor("data/tennis_dataset")
    
    # Load dataframes (assuming they exist)
    train_df = pd.read_pickle("data/processed/train_data.pkl")
    val_df = pd.read_pickle("data/processed/val_data.pkl")
    test_df = pd.read_pickle("data/processed/test_data.pkl")
    
    # Extract features
    X_train, y_train = processor.extract_features_for_classical_ml(train_df)
    X_val, y_val = processor.extract_features_for_classical_ml(val_df)
    X_test, y_test = processor.extract_features_for_classical_ml(test_df)
    
    # Initialize trainer
    trainer = ClassicalMLTrainer()
    
    # Prepare data
    X_train_scaled, X_val_scaled, X_test_scaled = trainer.prepare_data(X_train, X_val, X_test)
    
    # Train models
    trainer.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
    trainer.train_svm(X_train_scaled, y_train, X_val_scaled, y_val)
    trainer.train_logistic_regression(X_train_scaled, y_train, X_val_scaled, y_val)
    trainer.train_knn(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # Evaluate models
    test_results = trainer.evaluate_models(X_test_scaled, y_test)
    
    # Plot results
    trainer.plot_confusion_matrices(test_results, "results/confusion_matrices.png")
    trainer.plot_model_comparison(test_results, "results/model_comparison.png")
    
    # Save models
    trainer.save_models("models/classical_ml")


if __name__ == "__main__":
    main()