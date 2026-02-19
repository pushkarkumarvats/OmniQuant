"""
Boosting Alpha Model
Gradient boosting models (XGBoost, LightGBM, CatBoost) for alpha prediction
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from loguru import logger
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
import joblib


class BoostingAlphaModel:
    """Gradient boosting model for alpha prediction (XGBoost, LightGBM, CatBoost)."""
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        config: Optional[Dict[str, Any]] = None
    ):
        self.model_type = model_type.lower()
        self.config = config or {}
        self.model = None
        self.feature_importance = None
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                gamma=self.config.get('gamma', 0),
                reg_alpha=self.config.get('reg_alpha', 0),
                reg_lambda=self.config.get('reg_lambda', 1),
                random_state=42,
                n_jobs=-1
            )
            
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.get('n_estimators', 100),
                num_leaves=self.config.get('num_leaves', 31),
                max_depth=self.config.get('max_depth', -1),
                learning_rate=self.config.get('learning_rate', 0.05),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                reg_alpha=self.config.get('reg_alpha', 0),
                reg_lambda=self.config.get('reg_lambda', 0),
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
        elif self.model_type == 'catboost':
            self.model = CatBoostRegressor(
                iterations=self.config.get('n_estimators', 100),
                depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                l2_leaf_reg=self.config.get('reg_lambda', 3),
                random_state=42,
                verbose=False
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train the model and return metrics dict with R² scores."""
        logger.info(f"Training {self.model_type} model with {X_train.shape[1]} features")
        
        # Early stopping rounds
        early_stopping_rounds = self.config.get('early_stopping_rounds', 50)
        
        if self.model_type == 'xgboost':
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
        elif self.model_type == 'lightgbm':
            eval_set = [(X_train, y_train)]
            if X_val is not None and y_val is not None:
                eval_set.append((X_val, y_val))
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            
        elif self.model_type == 'catboost':
            eval_set = None
            if X_val is not None and y_val is not None:
                eval_set = (X_val, y_val)
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds,
                verbose=False
            )
        
        # Store feature importance
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'lightgbm':
            importance = self.model.feature_importances_
        else:  # catboost
            importance = self.model.feature_importances_
        
        if feature_names:
            self.feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
        else:
            self.feature_importance = pd.Series(importance).sort_values(ascending=False)
        
        # Calculate metrics
        train_score = self.model.score(X_train, y_train)
        metrics = {'train_r2': train_score}
        
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
            metrics['val_r2'] = val_score
            logger.info(f"Training completed - Train R²: {train_score:.4f}, Val R²: {val_score:.4f}")
        else:
            logger.info(f"Training completed - Train R²: {train_score:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None
    ) -> pd.Series:
        """Return feature importance as a sorted Series, optionally limited to top_n."""
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if top_n:
            return self.feature_importance.head(top_n)
        return self.feature_importance
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, float]:
        """Run k-fold cross-validation and return mean/std R² scores."""
        logger.info(f"Performing {cv}-fold cross-validation")
        
        scores = cross_val_score(
            self.model,
            X, y,
            cv=cv,
            scoring='r2',
            n_jobs=-1
        )
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        logger.info(f"CV R² Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        return cv_results
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Optional[Dict[str, List]] = None,
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """Optuna-based hyperparameter search. Retrains model with best params found."""
        import optuna
        from sklearn.model_selection import cross_val_score
        
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials")
        
        def objective(trial):
            # Define parameter search space
            if self.model_type == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
                }
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                
            elif self.model_type == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5)
                }
                model = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                
            else:  # catboost
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
                }
                model = CatBoostRegressor(**params, random_state=42, verbose=False)
            
            # Cross-validation score
            score = cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()
            return score
        
        # Optimize
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"Best R² score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Retrain with best parameters
        self.config.update(study.best_params)
        self._initialize_model()
        self.train(X_train, y_train)
        
        return {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'config': self.config,
            'feature_importance': self.feature_importance
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        
        self.model = data['model']
        self.model_type = data['model_type']
        self.config = data['config']
        self.feature_importance = data['feature_importance']
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_regression
    
    # Generate synthetic data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train models
    for model_type in ['xgboost', 'lightgbm', 'catboost']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing {model_type.upper()}")
        logger.info(f"{'='*50}")
        
        model = BoostingAlphaModel(model_type=model_type)
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Predictions
        predictions = model.predict(X_test)
        test_r2 = model.model.score(X_test, y_test)
        logger.info(f"Test R²: {test_r2:.4f}")
        
        # Feature importance
        top_features = model.get_feature_importance(top_n=5)
        logger.info(f"Top 5 features:\n{top_features}")
