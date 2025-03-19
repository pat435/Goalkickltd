"""
Model training module for the Goalkick Ltd Trading Bot.
Handles training and tuning of machine learning models for price prediction.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from config.logging_config import get_logger
from config.bot_config import PATHS
from src.utils.error_handling import handle_error

logger = get_logger("models.model_training")

class ModelTrainer:
    """Class for training and evaluating machine learning models."""
    
    def __init__(self, datastore=None):
        """
        Initialize the ModelTrainer.
        
        Args:
            datastore: DataStore instance (optional)
        """
        self.datastore = datastore
        self.models = {
            # Classification models
            'logistic': LogisticRegression,
            'random_forest_clf': RandomForestClassifier,
            'svm_clf': SVC,
            
            # Regression models
            'linear': LinearRegression,
            'random_forest_reg': RandomForestRegressor,
            'svm_reg': SVR,
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost_clf'] = xgb.XGBClassifier
            self.models['xgboost_reg'] = xgb.XGBRegressor
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(*PATHS['models'].split('/')), exist_ok=True)
    
    def train_model(self, X_train, y_train, model_type='random_forest_clf', params=None, cv=5):
        """
        Train a machine learning model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_type (str): Type of model to train
            params (dict): Model parameters
            cv (int): Number of cross-validation folds
            
        Returns:
            object: Trained model
        """
        try:
            if model_type not in self.models:
                logger.error(f"Unknown model type: {model_type}")
                return None
            
            logger.info(f"Training {model_type} model")
            
            # Get model class
            model_class = self.models[model_type]
            
            # Set default parameters if not provided
            if params is None:
                params = self._get_default_params(model_type)
            
            # Create and train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            logger.info(f"{model_type} model training completed")
            return model
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            handle_error(e, f"Failed to train {model_type} model")
            return None
    
    def tune_hyperparameters(self, X_train, y_train, model_type='random_forest_clf', param_grid=None):
        """
        Tune model hyperparameters using grid search.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            model_type (str): Type of model to tune
            param_grid (dict): Parameter grid for search
            
        Returns:
            tuple: (Best model, Best parameters, Best score)
        """
        try:
            if model_type not in self.models:
                logger.error(f"Unknown model type: {model_type}")
                return None, None, None
            
            logger.info(f"Tuning hyperparameters for {model_type} model")
            
            # Get model class
            model_class = self.models[model_type]
            
            # Set default parameter grid if not provided
            if param_grid is None:
                param_grid = self._get_default_param_grid(model_type)
                # Create base model
            model = model_class()
            
            # Use time series cross-validation for financial data
            tscv = TimeSeriesSplit(n_splits=5)
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tscv,
                scoring='accuracy' if model_type.endswith('clf') else 'neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            logger.info(f"Best {model_type} parameters: {best_params}")
            logger.info(f"Best {model_type} score: {best_score}")
            
            return best_model, best_params, best_score
        except Exception as e:
            logger.error(f"Error tuning {model_type} model: {e}")
            handle_error(e, f"Failed to tune {model_type} model")
            return None, None, None
    
    def evaluate_model(self, model, X_test, y_test, model_type='random_forest_clf'):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            model_type (str): Type of model
            
        Returns:
            dict: Evaluation metrics
        """
        try:
            logger.info(f"Evaluating {model_type} model")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics based on model type
            if model_type.endswith('clf'):
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
                
                logger.info(f"Classification metrics: accuracy={accuracy:.4f}, precision={precision:.4f}, "
                           f"recall={recall:.4f}, f1={f1:.4f}")
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                }
                
                logger.info(f"Regression metrics: MSE={mse:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_type} model: {e}")
            handle_error(e, f"Failed to evaluate {model_type} model")
            return {}
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            model_name (str): Name of the model
            metadata (dict): Additional model metadata
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Create model path
            model_dir = os.path.join(*PATHS['models'].split('/'))
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            
            # Create metadata dictionary
            if metadata is None:
                metadata = {}
            
            metadata['model_name'] = model_name
            metadata['created_at'] = datetime.now().isoformat()
            
            # Save model
            joblib.dump(model, model_path)
            
            # Save metadata
            meta_path = os.path.join(model_dir, f"{model_name}_metadata.json")
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model {model_name} saved to {model_path}")
            
            # Also save to datastore if available
            if self.datastore:
                self.datastore.save_model(model_name, model, metadata)
            
            return True
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e}")
            handle_error(e, f"Failed to save model {model_name}")
            return False
    
    def load_model(self, model_name):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            tuple: (Model, Metadata)
        """
        try:
            # Try to load from datastore first
            if self.datastore:
                model, metadata = self.datastore.load_model(model_name)
                if model is not None:
                    logger.info(f"Model {model_name} loaded from datastore")
                    return model, metadata
            
            # Fall back to loading from file
            model_dir = os.path.join(*PATHS['models'].split('/'))
            model_path = os.path.join(model_dir, f"{model_name}.pkl")
            meta_path = os.path.join(model_dir, f"{model_name}_metadata.json")
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model {model_name} not found at {model_path}")
                return None, None
            
            # Load model
            model = joblib.load(model_path)
            
            # Load metadata if available
            metadata = None
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            logger.info(f"Model {model_name} loaded from {model_path}")
            return model, metadata
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            handle_error(e, f"Failed to load model {model_name}")
            return None, None
    
    def _get_default_params(self, model_type):
        """
        Get default parameters for a model type.
        
        Args:
            model_type (str): Type of model
            
        Returns:
            dict: Default parameters
        """
        if model_type == 'logistic':
            return {'C': 1.0, 'max_iter': 1000, 'class_weight': 'balanced'}
        elif model_type == 'random_forest_clf':
            return {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'}
        elif model_type == 'svm_clf':
            return {'C': 1.0, 'kernel': 'rbf', 'class_weight': 'balanced'}
        elif model_type == 'linear':
            return {}
        elif model_type == 'random_forest_reg':
            return {'n_estimators': 100, 'max_depth': 10}
        elif model_type == 'svm_reg':
            return {'C': 1.0, 'kernel': 'rbf'}
        elif model_type == 'xgboost_clf' and XGBOOST_AVAILABLE:
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'scale_pos_weight': 1}
        elif model_type == 'xgboost_reg' and XGBOOST_AVAILABLE:
            return {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1}
        else:
            return {}
    
    def _get_default_param_grid(self, model_type):
        """
        Get default parameter grid for a model type.
        
        Args:
            model_type (str): Type of model
            
        Returns:
            dict: Default parameter grid
        """
        if model_type == 'logistic':
            return {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000],
                'class_weight': ['balanced', None]
            }
        elif model_type == 'random_forest_clf':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
        elif model_type == 'svm_clf':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'class_weight': ['balanced', None]
            }
        elif model_type == 'linear':
            return {}
        elif model_type == 'random_forest_reg':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'svm_reg':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'epsilon': [0.1, 0.2, 0.5]
            }
        elif model_type == 'xgboost_clf' and XGBOOST_AVAILABLE:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
                'scale_pos_weight': [1, 2, 5]
            }
        elif model_type == 'xgboost_reg' and XGBOOST_AVAILABLE:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:
            return {}