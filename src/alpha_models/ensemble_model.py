"""
Ensemble Alpha Model
Combines multiple alpha models using stacking, blending, or Bayesian averaging
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from loguru import logger
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import StackingRegressor
import joblib


class EnsembleAlphaModel:
    """
    Ensemble model that combines multiple alpha models
    """
    
    def __init__(
        self,
        method: str = 'stacking',
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Ensemble Alpha Model
        
        Args:
            method: 'stacking', 'blending', 'weighted_average', or 'bayesian'
            config: Model configuration
        """
        self.method = method.lower()
        self.config = config or {}
        self.base_models = []
        self.meta_model = None
        self.weights = None
        
    def add_model(self, name: str, model: Any):
        """
        Add a base model to the ensemble
        
        Args:
            name: Model name
            model: Model instance (must have fit and predict methods)
        """
        self.base_models.append({
            'name': name,
            'model': model
        })
        logger.info(f"Added model: {name}")
    
    def train_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        meta_model: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble using stacking
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            meta_model: Meta-learner (default: Ridge)
            
        Returns:
            Training metrics
        """
        logger.info("Training stacking ensemble")
        
        if not self.base_models:
            raise ValueError("No base models added. Call add_model() first.")
        
        # Create stacking regressor
        estimators = [(model['name'], model['model']) for model in self.base_models]
        
        if meta_model is None:
            meta_model = Ridge(alpha=1.0)
        
        self.meta_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=5
        )
        
        # Train
        self.meta_model.fit(X_train, y_train)
        
        # Calculate metrics
        train_score = self.meta_model.score(X_train, y_train)
        metrics = {'train_r2': train_score}
        
        if X_val is not None and y_val is not None:
            val_score = self.meta_model.score(X_val, y_val)
            metrics['val_r2'] = val_score
            logger.info(f"Stacking - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
        else:
            logger.info(f"Stacking - Train R²: {train_score:.4f}")
        
        return metrics
    
    def train_blending(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_blend: np.ndarray,
        y_blend: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train ensemble using blending
        
        Args:
            X_train: Training features for base models
            y_train: Training targets
            X_blend: Blending set features
            y_blend: Blending set targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training metrics
        """
        logger.info("Training blending ensemble")
        
        if not self.base_models:
            raise ValueError("No base models added. Call add_model() first.")
        
        # Train base models on training set
        for model_dict in self.base_models:
            logger.info(f"Training base model: {model_dict['name']}")
            model_dict['model'].fit(X_train, y_train)
        
        # Generate predictions on blending set
        blend_predictions = []
        for model_dict in self.base_models:
            preds = model_dict['model'].predict(X_blend)
            blend_predictions.append(preds)
        
        X_blend_meta = np.column_stack(blend_predictions)
        
        # Train meta-model on blending set
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(X_blend_meta, y_blend)
        
        # Calculate metrics
        blend_score = self.meta_model.score(X_blend_meta, y_blend)
        metrics = {'blend_r2': blend_score}
        
        if X_val is not None and y_val is not None:
            # Generate predictions for validation set
            val_predictions = []
            for model_dict in self.base_models:
                preds = model_dict['model'].predict(X_val)
                val_predictions.append(preds)
            
            X_val_meta = np.column_stack(val_predictions)
            val_score = self.meta_model.score(X_val_meta, y_val)
            metrics['val_r2'] = val_score
            logger.info(f"Blending - Blend R²: {blend_score:.4f}, Val R²: {val_score:.4f}")
        else:
            logger.info(f"Blending - Blend R²: {blend_score:.4f}")
        
        return metrics
    
    def train_weighted_average(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimization_method: str = 'inverse_mse'
    ) -> Dict[str, Any]:
        """
        Train weighted average ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimization_method: 'inverse_mse', 'inverse_variance', or 'optimize'
            
        Returns:
            Training metrics and weights
        """
        logger.info(f"Training weighted average ensemble ({optimization_method})")
        
        if not self.base_models:
            raise ValueError("No base models added. Call add_model() first.")
        
        # Train base models
        predictions_val = []
        mse_scores = []
        
        for model_dict in self.base_models:
            logger.info(f"Training base model: {model_dict['name']}")
            model_dict['model'].fit(X_train, y_train)
            
            # Validation predictions
            preds = model_dict['model'].predict(X_val)
            predictions_val.append(preds)
            
            # Calculate MSE
            mse = np.mean((preds - y_val) ** 2)
            mse_scores.append(mse)
        
        predictions_val = np.array(predictions_val)
        
        # Calculate weights
        if optimization_method == 'inverse_mse':
            # Weights inversely proportional to MSE
            inverse_mse = 1 / (np.array(mse_scores) + 1e-10)
            self.weights = inverse_mse / inverse_mse.sum()
            
        elif optimization_method == 'inverse_variance':
            # Weights inversely proportional to prediction variance
            variances = np.var(predictions_val, axis=1)
            inverse_var = 1 / (variances + 1e-10)
            self.weights = inverse_var / inverse_var.sum()
            
        elif optimization_method == 'optimize':
            # Optimize weights using scipy
            from scipy.optimize import minimize
            
            def objective(weights):
                weighted_pred = np.dot(weights, predictions_val)
                return np.mean((weighted_pred - y_val) ** 2)
            
            # Constraints: weights sum to 1 and are non-negative
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(len(self.base_models))]
            
            # Initial weights (equal)
            w0 = np.ones(len(self.base_models)) / len(self.base_models)
            
            result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)
            self.weights = result.x
        
        # Calculate ensemble predictions
        ensemble_pred = np.dot(self.weights, predictions_val)
        ensemble_mse = np.mean((ensemble_pred - y_val) ** 2)
        ensemble_r2 = 1 - (np.sum((y_val - ensemble_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
        
        logger.info(f"Model weights: {dict(zip([m['name'] for m in self.base_models], self.weights))}")
        logger.info(f"Ensemble R²: {ensemble_r2:.4f}, MSE: {ensemble_mse:.6f}")
        
        return {
            'weights': self.weights,
            'val_r2': ensemble_r2,
            'val_mse': ensemble_mse,
            'individual_mse': dict(zip([m['name'] for m in self.base_models], mse_scores))
        }
    
    def train_bayesian(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Train Bayesian Model Averaging ensemble
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training metrics
        """
        logger.info("Training Bayesian Model Averaging ensemble")
        
        if not self.base_models:
            raise ValueError("No base models added. Call add_model() first.")
        
        # Train base models and calculate likelihoods
        log_likelihoods = []
        predictions_val = []
        
        for model_dict in self.base_models:
            logger.info(f"Training base model: {model_dict['name']}")
            model_dict['model'].fit(X_train, y_train)
            
            # Validation predictions
            preds = model_dict['model'].predict(X_val)
            predictions_val.append(preds)
            
            # Calculate log likelihood (assuming Gaussian)
            residuals = y_val - preds
            sigma = np.std(residuals)
            log_likelihood = -0.5 * len(y_val) * np.log(2 * np.pi * sigma**2) - np.sum(residuals**2) / (2 * sigma**2)
            log_likelihoods.append(log_likelihood)
        
        # Convert to posterior probabilities
        log_likelihoods = np.array(log_likelihoods)
        max_ll = np.max(log_likelihoods)
        exp_ll = np.exp(log_likelihoods - max_ll)  # Subtract max for numerical stability
        self.weights = exp_ll / exp_ll.sum()
        
        # Calculate ensemble predictions
        predictions_val = np.array(predictions_val)
        ensemble_pred = np.dot(self.weights, predictions_val)
        ensemble_r2 = 1 - (np.sum((y_val - ensemble_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))
        
        logger.info(f"Posterior probabilities: {dict(zip([m['name'] for m in self.base_models], self.weights))}")
        logger.info(f"Ensemble R²: {ensemble_r2:.4f}")
        
        return {
            'weights': self.weights,
            'val_r2': ensemble_r2,
            'log_likelihoods': dict(zip([m['name'] for m in self.base_models], log_likelihoods))
        }
    
    def train(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Train ensemble using the specified method
        
        Returns:
            Training metrics
        """
        if self.method == 'stacking':
            return self.train_stacking(*args, **kwargs)
        elif self.method == 'blending':
            return self.train_blending(*args, **kwargs)
        elif self.method == 'weighted_average':
            return self.train_weighted_average(*args, **kwargs)
        elif self.method == 'bayesian':
            return self.train_bayesian(*args, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Model not trained. Call train() first.")
            return self.meta_model.predict(X)
            
        elif self.method in ['blending', 'weighted_average', 'bayesian']:
            if not self.base_models or self.weights is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Generate predictions from all base models
            predictions = []
            for model_dict in self.base_models:
                preds = model_dict['model'].predict(X)
                predictions.append(preds)
            
            predictions = np.array(predictions)
            
            if self.method == 'blending' and self.meta_model is not None:
                # Use meta-model
                X_meta = predictions.T
                return self.meta_model.predict(X_meta)
            else:
                # Weighted average
                return np.dot(self.weights, predictions)
    
    def get_model_contributions(self, X: np.ndarray) -> pd.DataFrame:
        """
        Get individual model contributions to ensemble prediction
        
        Args:
            X: Input features
            
        Returns:
            DataFrame with model predictions and weights
        """
        if not self.base_models:
            raise ValueError("No models in ensemble")
        
        contributions = {}
        
        for i, model_dict in enumerate(self.base_models):
            preds = model_dict['model'].predict(X)
            contributions[model_dict['name']] = preds
            
            if self.weights is not None:
                contributions[f"{model_dict['name']}_weight"] = self.weights[i]
        
        df = pd.DataFrame(contributions)
        df['ensemble_prediction'] = self.predict(X)
        
        return df
    
    def save_model(self, filepath: str):
        """
        Save ensemble model to file
        
        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'method': self.method,
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'weights': self.weights,
            'config': self.config
        }, filepath)
        
        logger.info(f"Ensemble model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load ensemble model from file
        
        Args:
            filepath: Path to model file
        """
        data = joblib.load(filepath)
        
        self.method = data['method']
        self.base_models = data['base_models']
        self.meta_model = data['meta_model']
        self.weights = data['weights']
        self.config = data['config']
        
        logger.info(f"Ensemble model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Test different ensemble methods
    for method in ['weighted_average', 'stacking']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {method.upper()}")
        logger.info(f"{'='*50}")
        
        # Create ensemble
        ensemble = EnsembleAlphaModel(method=method)
        
        # Add base models
        ensemble.add_model('linear', LinearRegression())
        ensemble.add_model('tree', DecisionTreeRegressor(max_depth=5, random_state=42))
        ensemble.add_model('forest', RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42))
        
        # Train
        if method == 'weighted_average':
            metrics = ensemble.train(X_train, y_train, X_val, y_val, optimization_method='optimize')
        else:
            metrics = ensemble.train(X_train, y_train, X_val, y_val)
        
        # Predict
        predictions = ensemble.predict(X_test)
        test_r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        logger.info(f"Test R²: {test_r2:.4f}")
