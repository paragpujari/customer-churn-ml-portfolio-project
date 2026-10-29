"""
End-to-End Training Script for Customer Churn Prediction
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_clean_data(path):
    """Load and perform basic cleaning"""
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True)}")
    
    # Convert TotalCharges to numeric (has some empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop rows with missing TotalCharges (very few)
    df = df.dropna(subset=['TotalCharges'])
    
    # Drop customerID (not useful for prediction)
    df = df.drop('customerID', axis=1)
    
    # Convert SeniorCitizen to object for consistency (it's 0/1)
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
    
    # Target encoding
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    print(f"After cleaning: {df.shape}")
    return df


def get_feature_lists(df):
    """Identify numeric and categorical features"""
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in df.columns if col not in numeric_features + ['Churn']]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features):
    """Create ColumnTransformer for preprocessing"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def evaluate_model(model, X_test, y_test, model_name):
    """Print evaluation metrics"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"\n{'='*50}")
    print(f"Evaluation for {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")
    if y_proba is not None:
        print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }


def main():
    print("="*60)
    print("CUSTOMER CHURN PREDICTION - END TO END TRAINING")
    print("="*60)
    
    # 1. Load & Clean
    df = load_and_clean_data(DATA_PATH)
    
    # 2. Features
    numeric_features, categorical_features = get_feature_lists(df)
    print(f"\nNumeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 4. Preprocessor
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    # 5. Models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, 
            class_weight='balanced', n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, clf in models.items():
        print(f"\n>>> Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, name)
        results[name] = metrics
        
        # Use F1 as primary metric for imbalanced data
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            best_model = pipeline
            best_name = name
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (F1 = {best_score:.4f})")
    print(f"{'='*60}")
    
    # 6. Save best model + metadata
    model_path = os.path.join(MODEL_DIR, 'churn_pipeline.joblib')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Also save feature lists for the app
    meta = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'best_model_name': best_name,
        'metrics': results[best_name]
    }
    meta_path = os.path.join(MODEL_DIR, 'model_meta.joblib')
    joblib.dump(meta, meta_path)
    print(f"Metadata saved to: {meta_path}")
    
    # Feature importance for Random Forest / GB if applicable
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        # Get feature names after preprocessing
        try:
            ohe = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(categorical_features).tolist()
            all_feature_names = numeric_features + cat_feature_names
            importances = best_model.named_steps['classifier'].feature_importances_
            fi = pd.Series(importances, index=all_feature_names).sort_values(ascending=False)
            print("\nTop 15 Feature Importances:")
            print(fi.head(15))
            fi.to_csv(os.path.join(MODEL_DIR, 'feature_importances.csv'))
        except Exception as e:
            print(f"Could not extract feature importances: {e}")
    
    print("\n✅ Training completed successfully!")
    return best_model


if __name__ == "__main__":
    main()
