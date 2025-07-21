import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import mlflow
import mlflow.sklearn
from src.data.data_processing import process_data
from src.utils.logger import get_logger
import os
import numpy as np

logger = get_logger()

FEATURES = [
    'age', 'gender', 'marital_status', 'residential_status', 'employment_stats', 'monthly_income',
    'other_income_source', 'total_monthly_expenses', 'income_to_expense_ratio', 'number_of_loans',
    'cibil_score', 'loan_amount_requested', 'loan_purpose', 'loan_term_in_months',
    'location_mismatch', 'document_verification_score'
]

def train_individual_models(X_train, y_train, X_val, y_val):
    """Train individual models and return their performance metrics"""
    models = {}
    metrics = {}
    
    #TODO: Train each model with different hyperparameters to evaluate which set of hyperparameter works best 
    #TODO: Log each version of the model with its metrics to MLflow
    
    # LightGBM
    logger.info('Training LightGBM...')
    lgb_model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = lgb_model.predict(X_val)
    y_proba_lgb = lgb_model.predict_proba(X_val)[:, 1]
    
    models['lightgbm'] = lgb_model
    metrics['lightgbm'] = {
        'accuracy': accuracy_score(y_val, y_pred_lgb),
        'precision': precision_score(y_val, y_pred_lgb, zero_division=0),
        'recall': recall_score(y_val, y_pred_lgb, zero_division=0),
        'f1': f1_score(y_val, y_pred_lgb, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_proba_lgb)
    }
    
    # XGBoost
    logger.info('Training XGBoost...')
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
    
    models['xgboost'] = xgb_model
    metrics['xgboost'] = {
        'accuracy': accuracy_score(y_val, y_pred_xgb),
        'precision': precision_score(y_val, y_pred_xgb, zero_division=0),
        'recall': recall_score(y_val, y_pred_xgb, zero_division=0),
        'f1': f1_score(y_val, y_pred_xgb, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_proba_xgb)
    }
    
    # Random Forest
    logger.info('Training Random Forest...')
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_val)
    y_proba_rf = rf_model.predict_proba(X_val)[:, 1]
    
    models['random_forest'] = rf_model
    metrics['random_forest'] = {
        'accuracy': accuracy_score(y_val, y_pred_rf),
        'precision': precision_score(y_val, y_pred_rf, zero_division=0),
        'recall': recall_score(y_val, y_pred_rf, zero_division=0),
        'f1': f1_score(y_val, y_pred_rf, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_proba_rf)
    }
    
    return models, metrics

def create_ensemble(models, X_val, y_val):
    """Create and evaluate ensemble model"""
    logger.info('Creating ensemble model...')
    
    # Voting Classifier (soft voting)
    ensemble = VotingClassifier(
        estimators=[
            ('lightgbm', models['lightgbm']),
            ('xgboost', models['xgboost']),
            ('random_forest', models['random_forest'])
        ],
        voting='soft'
    )
    
    # Fit ensemble on validation data to get predictions
    ensemble.fit(X_val, y_val)  # This is just for evaluation
    y_pred_ensemble = ensemble.predict(X_val)
    y_proba_ensemble = ensemble.predict_proba(X_val)[:, 1]
    
    ensemble_metrics = {
        'accuracy': accuracy_score(y_val, y_pred_ensemble),
        'precision': precision_score(y_val, y_pred_ensemble, zero_division=0),
        'recall': recall_score(y_val, y_pred_ensemble, zero_division=0),
        'f1': f1_score(y_val, y_pred_ensemble, zero_division=0),
        'roc_auc': roc_auc_score(y_val, y_proba_ensemble)
    }
    
    return ensemble, ensemble_metrics

def train_fraud_model(train_csv_path, model_save_path):
    """
    Train an ensemble of models for fraud detection with cross-validation and hyperparameter tuning.
    """
    logger.info('Loading training data...')
    try:
        df = pd.read_csv(train_csv_path)
        logger.info(f'Data shape: {df.shape}')
    except Exception as e:
        logger.error(f'Error loading training data: {e}')
        return

    try:
        logger.info('Processing data...')
        df = process_data(df, is_train=True)
        
        X = df[FEATURES]
        y = df['is_fraud']
        
        #TODO: Split data into train, validation, and test sets, 60% train, 20% validation, 20% test and also stratify the data on 'is_fraud'
        
        #TODO: Apply SMOTE to balance the training data - since fraud data is very less
        
        # Train individual models
        models, metrics = train_individual_models(X_train_balanced, y_train_balanced, X_val, y_val)
        
        # Create ensemble
        ensemble, ensemble_metrics = create_ensemble(models, X_val, y_val)
        
        #TODO: Save the ensemble model to models folder
        #TODO: Log all metrics to MLflow
        #TODO: Save the ensemble and individual models to mlflow registry
        
        #TODO: Run the model on test set and log the metrics to MLflow
        
      
        
    except Exception as e:
        logger.error(f'Error training model: {e}')
        return

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--model_path', type=str, default='models/fraud_model_ensemble.pkl', help='Path to save model')
    args = parser.parse_args()
    train_fraud_model(args.train_csv, args.model_path) 