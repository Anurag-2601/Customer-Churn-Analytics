"""
Train and evaluate multiple machine learning models for customer churn prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import xgboost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Skipping XGBoost model.")

class ChurnModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test):
        """Initialize model trainer with train/test data"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)
        
        self.models['Logistic Regression'] = model
        self._evaluate_model('Logistic Regression', model)
        
    def train_random_forest(self):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        self.models['Random Forest'] = model
        self._evaluate_model('Random Forest', model)
        
    def train_gradient_boosting(self):
        """Train Gradient Boosting model"""
        print("\n" + "="*50)
        print("Training Gradient Boosting...")
        print("="*50)
        
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['Gradient Boosting'] = model
        self._evaluate_model('Gradient Boosting', model)
        
    def train_xgboost(self):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            print("\nXGBoost not available. Skipping...")
            return
            
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        
        model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        model.fit(self.X_train, self.y_train)
        
        self.models['XGBoost'] = model
        self._evaluate_model('XGBoost', model)
        
    def _evaluate_model(self, model_name, model):
        """Evaluate model performance"""
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Cross-validation Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        
    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results],
            'Precision': [self.results[m]['precision'] for m in self.results],
            'Recall': [self.results[m]['recall'] for m in self.results],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results],
            'ROC-AUC': [self.results[m]['roc_auc'] for m in self.results]
        })
        
        comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_df.to_csv('../output/model_comparison.csv', index=False)
        
        return comparison_df
        
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name in self.results:
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = self.results[model_name]['roc_auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('../output/roc_curves.png', dpi=300, bbox_inches='tight')
        print("\nROC curves saved to output/roc_curves.png")
        plt.close()
        
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(self.results):
            y_pred = self.results[model_name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'],
                       ax=axes[idx], cbar_kws={'shrink': 0.8})
            axes[idx].set_title(f'{model_name}', fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('../output/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved to output/confusion_matrices.png")
        plt.close()
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        tree_models = ['Random Forest', 'Gradient Boosting']
        if XGBOOST_AVAILABLE:
            tree_models.append('XGBoost')
        
        available_tree_models = [m for m in tree_models if m in self.models]
        
        if not available_tree_models:
            print("\nNo tree-based models available for feature importance.")
            return
        
        n_models = len(available_tree_models)
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(available_tree_models):
            model = self.models[model_name]
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create DataFrame and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True).tail(15)
            
            # Plot
            axes[idx].barh(range(len(importance_df)), importance_df['importance'])
            axes[idx].set_yticks(range(len(importance_df)))
            axes[idx].set_yticklabels(importance_df['feature'])
            axes[idx].set_xlabel('Importance', fontsize=10)
            axes[idx].set_title(f'{model_name}\nTop 15 Features', fontweight='bold')
            axes[idx].grid(alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('../output/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to output/feature_importance.png")
        plt.close()
        
    def save_best_model(self):
        """Save the best performing model"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        best_model = self.models[best_model_name]
        
        joblib.dump(best_model, '../models/best_model.pkl')
        
        print(f"\nBest model ({best_model_name}) saved to models/best_model.pkl")
        print(f"Best model ROC-AUC: {self.results[best_model_name]['roc_auc']:.4f}")
        
        return best_model_name


if __name__ == "__main__":
    # This script should be run after preprocessing
    # For demonstration, we'll assume the data has been processed
    from preprocess_data import ChurnDataPreprocessor
    
    print("="*70)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING")
    print("="*70)
    
    # Load and preprocess data
    preprocessor = ChurnDataPreprocessor('../data/customer_churn_data.csv')
    preprocessor.handle_missing_values()
    preprocessor.engineer_features()
    preprocessor.encode_categorical_features()
    X_train, X_test, y_train, y_test = preprocessor.prepare_for_modeling()
    
    # Initialize trainer
    trainer = ChurnModelTrainer(X_train, X_test, y_train, y_test)
    
    # Train models
    trainer.train_logistic_regression()
    trainer.train_random_forest()
    trainer.train_gradient_boosting()
    trainer.train_xgboost()
    
    # Compare models
    comparison = trainer.compare_models()
    
    # Generate visualizations
    trainer.plot_roc_curves()
    trainer.plot_confusion_matrices()
    trainer.plot_feature_importance()
    
    # Save best model
    best_model = trainer.save_best_model()
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
