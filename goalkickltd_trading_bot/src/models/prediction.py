"""
Prediction module for the Goalkick Ltd Trading Bot.
Handles making predictions using trained models.
"""

import pandas as pd
import numpy as np
from collections import Counter

from config.logging_config import get_logger
from src.models.model_training import ModelTrainer
from src.models.feature_engineering import FeatureEngineer
from src.utils.error_handling import handle_error

logger = get_logger("models.prediction")

class PredictionEngine:
    """Class for making predictions using trained models."""
    
    def __init__(self, datastore=None):
        """
        Initialize the PredictionEngine.
        
        Args:
            datastore: DataStore instance (optional)
        """
        self.model_trainer = ModelTrainer(datastore)
        self.feature_engineer = FeatureEngineer()
        self.loaded_models = {}  # model_name -> (model, metadata)
    
    def load_model(self, model_name):
        """
        Load a model for prediction.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if model_name in self.loaded_models:
                logger.debug(f"Model {model_name} already loaded")
                return True
            
            # Load model
            model, metadata = self.model_trainer.load_model(model_name)
            
            if model is None:
                logger.error(f"Failed to load model {model_name}")
                return False
            
            # Store model and metadata
            self.loaded_models[model_name] = (model, metadata)
            
            logger.info(f"Model {model_name} loaded for prediction")
            return True
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            handle_error(e, f"Failed to load model {model_name}")
            return False
    
    def unload_model(self, model_name):
        """
        Unload a model from memory.
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            bool: True if unloaded successfully, False otherwise
        """
        try:
            if model_name in self.loaded_models:
                self.loaded_models.pop(model_name)
                logger.info(f"Model {model_name} unloaded")
                return True
            else:
                logger.warning(f"Model {model_name} not loaded")
                return False
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            handle_error(e, f"Failed to unload model {model_name}")
            return False
    
    def predict(self, model_name, data, feature_sets=None, normalize=True):
        """
        Make predictions using a loaded model.
        
        Args:
            model_name (str): Name of the model
            data (pd.DataFrame): Data to make predictions on
            feature_sets (list): Feature sets to generate
            normalize (bool): Whether to normalize features
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            # Ensure model is loaded
            if model_name not in self.loaded_models:
                if not self.load_model(model_name):
                    logger.error(f"Failed to load model {model_name}")
                    return None
            
            model, metadata = self.loaded_models[model_name]
            
            # Generate features
            features_df = self.feature_engineer.generate_features(data, feature_sets)
            
            if features_df.empty:
                logger.error("Failed to generate features")
                return None
            
            # Normalize features if requested
            if normalize:
                features_df = self.feature_engineer.normalize_features(features_df)
            
            # Get required features from metadata if available
            required_features = None
            if metadata and 'features' in metadata:
                required_features = metadata['features']
                
                # Ensure all required features are present
                missing_features = [f for f in required_features if f not in features_df.columns]
                if missing_features:
                    logger.warning(f"Missing required features: {missing_features}")
                    # We can still continue with available features
            
            # Select features or use all available
            if required_features:
                available_features = [f for f in required_features if f in features_df.columns]
                X = features_df[available_features]
            else:
                # Exclude OHLCV columns
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                feature_cols = [col for col in features_df.columns if col not in ohlcv_cols]
                X = features_df[feature_cols]
            
            # Handle NaN values
            X = X.fillna(0)
            
            # Make predictions
            predictions = model.predict(X)
            
            logger.debug(f"Made {len(predictions)} predictions with model {model_name}")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions with model {model_name}: {e}")
            handle_error(e, f"Failed to make predictions with model {model_name}")
            return None
    
    def predict_proba(self, model_name, data, feature_sets=None, normalize=True):
        """
        Make probability predictions using a loaded classification model.
        
        Args:
            model_name (str): Name of the model
            data (pd.DataFrame): Data to make predictions on
            feature_sets (list): Feature sets to generate
            normalize (bool): Whether to normalize features
            
        Returns:
            np.ndarray: Probability predictions
        """
        try:
            # Ensure model is loaded
            if model_name not in self.loaded_models:
                if not self.load_model(model_name):
                    logger.error(f"Failed to load model {model_name}")
                    return None
            
            model, metadata = self.loaded_models[model_name]
            
            # Check if model supports predict_proba
            if not hasattr(model, 'predict_proba'):
                logger.error(f"Model {model_name} does not support probability predictions")
                return None
            
            # Generate features
            features_df = self.feature_engineer.generate_features(data, feature_sets)
            
            if features_df.empty:
                logger.error("Failed to generate features")
                return None
            
            # Normalize features if requested
            if normalize:
                features_df = self.feature_engineer.normalize_features(features_df)
            
            # Get required features from metadata if available
            required_features = None
            if metadata and 'features' in metadata:
                required_features = metadata['features']
                
                # Ensure all required features are present
                missing_features = [f for f in required_features if f not in features_df.columns]
                if missing_features:
                    logger.warning(f"Missing required features: {missing_features}")
            
            # Select features or use all available
            if required_features:
                available_features = [f for f in required_features if f in features_df.columns]
                X = features_df[available_features]
            else:
                # Exclude OHLCV columns
                ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                feature_cols = [col for col in features_df.columns if col not in ohlcv_cols]
                X = features_df[feature_cols]
            
            # Handle NaN values
            X = X.fillna(0)
            
            # Make probability predictions
            probabilities = model.predict_proba(X)
            
            logger.debug(f"Made probability predictions with model {model_name}")
            return probabilities
        except Exception as e:
            logger.error(f"Error making probability predictions with model {model_name}: {e}")
            handle_error(e, f"Failed to make probability predictions with model {model_name}")
            return None
    
    def predict_ensemble(self, model_names, data, feature_sets=None, weights=None, normalize=True):
        """
        Make predictions using an ensemble of models.
        
        Args:
            model_names (list): List of model names
            data (pd.DataFrame): Data to make predictions on
            feature_sets (list): Feature sets to generate
            weights (list): Weights for each model
            normalize (bool): Whether to normalize features
            
        Returns:
            np.ndarray: Ensemble predictions
        """
        try:
            if not model_names:
                logger.error("No models specified for ensemble prediction")
                return None
            
            # Standardize weights
            if weights is None:
                weights = [1.0] * len(model_names)
            elif len(weights) != len(model_names):
                logger.warning("Mismatch between number of models and weights")
                weights = [1.0] * len(model_names)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1.0] * len(model_names)
                total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Generate features
            features_df = self.feature_engineer.generate_features(data, feature_sets)
            
            if features_df.empty:
                logger.error("Failed to generate features")
                return None
            
            # Normalize features if requested
            if normalize:
                features_df = self.feature_engineer.normalize_features(features_df)
            
            all_predictions = []
            all_probabilities = []
            is_classification = False
            
            # Make predictions with each model
            for i, model_name in enumerate(model_names):
                # Ensure model is loaded
                if model_name not in self.loaded_models:
                    if not self.load_model(model_name):
                        logger.warning(f"Failed to load model {model_name}, skipping")
                        continue
                
                model, metadata = self.loaded_models[model_name]
                
                # Get required features from metadata if available
                required_features = None
                if metadata and 'features' in metadata:
                    required_features = metadata['features']
                    
                    # Ensure all required features are present
                    missing_features = [f for f in required_features if f not in features_df.columns]
                    if missing_features:
                        logger.warning(f"Missing required features for model {model_name}: {missing_features}")
                
                # Select features or use all available
                if required_features:
                    available_features = [f for f in required_features if f in features_df.columns]
                    X = features_df[available_features]
                else:
                    # Exclude OHLCV columns
                    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
                    feature_cols = [col for col in features_df.columns if col not in ohlcv_cols]
                    X = features_df[feature_cols]
                
                # Handle NaN values
                X = X.fillna(0)
                
                # Check if classification or regression
                if metadata and 'type' in metadata:
                    is_classification = metadata['type'] == 'classification'
                else:
                    # Infer from model type
                    model_type = model.__class__.__name__
                    is_classification = 'Classifier' in model_type or model_type in ['LogisticRegression', 'SVC']
                
                # Make predictions
                predictions = model.predict(X)
                all_predictions.append((predictions, weights[i]))
                
                # Get probabilities for classification models
                if is_classification and hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    all_probabilities.append((probabilities, weights[i]))
            
            if not all_predictions:
                logger.error("No predictions made by any model")
                return None
            
            # Combine predictions
            if is_classification:
                # For classification, use majority vote or probability averaging
                if all_probabilities:
                    # Combine probabilities
                    combined_proba = np.zeros(all_probabilities[0][0].shape)
                    for proba, weight in all_probabilities:
                        combined_proba += proba * weight
                    
                    # Get class with highest probability
                    final_predictions = np.argmax(combined_proba, axis=1)
                else:
                    # Simple majority vote
                    votes = []
                    for preds, weight in all_predictions:
                        for _ in range(int(weight * 10)):  # Convert weight to integer for voting
                            votes.append(preds)
                    
                    votes = np.array(votes)
                    final_predictions = []
                    
                    for i in range(len(votes[0])):
                        # Get most common prediction for this sample
                        sample_votes = votes[:, i]
                        most_common = Counter(sample_votes).most_common(1)[0][0]
                        final_predictions.append(most_common)
                    
                    final_predictions = np.array(final_predictions)
            else:
                # For regression, use weighted average
                final_predictions = np.zeros(all_predictions[0][0].shape)
                for preds, weight in all_predictions:
                    final_predictions += preds * weight
            
            logger.debug(f"Made ensemble predictions using {len(model_names)} models")
            return final_predictions
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            handle_error(e, f"Failed to make ensemble predictions")
            return None
    
    def get_prediction_confidence(self, model_name, data, feature_sets=None, normalize=True):
        """
        Get prediction confidence scores for a classification model.
        
        Args:
            model_name (str): Name of the model
            data (pd.DataFrame): Data to make predictions on
            feature_sets (list): Feature sets to generate
            normalize (bool): Whether to normalize features
            
        Returns:
            np.ndarray: Confidence scores
        """
        try:
            # Get probability predictions
            probabilities = self.predict_proba(model_name, data, feature_sets, normalize)
            
            if probabilities is None:
                return None
            
            # Calculate confidence as maximum probability
            confidence = np.max(probabilities, axis=1)
            
            return confidence
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            handle_error(e, f"Failed to calculate prediction confidence")
            return None
    
    def get_feature_importance(self, model_name):
        """
        Get feature importances for a model (if supported).
        
        Args:
            model_name (str): Name of the model
            
        Returns:
            dict: Feature importances (feature name -> importance)
        """
        try:
            # Ensure model is loaded
            if model_name not in self.loaded_models:
                if not self.load_model(model_name):
                    logger.error(f"Failed to load model {model_name}")
                    return None
            
            model, metadata = self.loaded_models[model_name]
            
            # Check if model supports feature importance
            if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
                logger.warning(f"Model {model_name} does not support feature importance")
                return None
            
            # Get feature names from metadata
            feature_names = metadata.get('features', [])
            
            # Get feature importances
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_).flatten()
            else:
                return None
            
            # Create feature importance dictionary
            if feature_names and len(feature_names) == len(importances):
                feature_importance = {feature_names[i]: importances[i] for i in range(len(feature_names))}
            else:
                feature_importance = {f"feature_{i}": importances[i] for i in range(len(importances))}
            
            # Sort by importance (descending)
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            return feature_importance
        except Exception as e:
            logger.error(f"Error getting feature importance for model {model_name}: {e}")
            handle_error(e, f"Failed to get feature importance for model {model_name}")
            return None