"""
Alpha Research Example
Demonstrates feature engineering and alpha model training
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from loguru import logger

from src.data_pipeline.ingestion import DataIngestion
from src.feature_engineering.microstructure_features import MicrostructureFeatures
from src.feature_engineering.technical_features import TechnicalFeatures
from src.alpha_models.boosting_model import BoostingAlphaModel
from src.alpha_models.lstm_model import LSTMAlphaModel

# 1. Generate synthetic data
logger.info("="*60)
logger.info("STEP 1: Data Generation")
logger.info("="*60)

ingestion = DataIngestion()
tick_data = ingestion.generate_synthetic_tick_data(num_ticks=10000, seed=42)
logger.info(f"Generated {len(tick_data)} tick records")

# 2. Feature engineering
logger.info("\n" + "="*60)
logger.info("STEP 2: Feature Engineering")
logger.info("="*60)

# Microstructure features
micro_features = MicrostructureFeatures()
data_with_micro = micro_features.generate_all_features(tick_data)

# Technical features (resample to bars first)
df_bars = tick_data.set_index('timestamp').resample('1min').agg({
    'price': 'last',
    'volume': 'sum',
    'bid': 'last',
    'ask': 'last'
}).dropna().reset_index()

tech_features = TechnicalFeatures()
data_with_features = tech_features.generate_all_features(df_bars, price_col='price')

logger.info(f"Total features: {len(data_with_features.columns)}")

# 3. Prepare training data
logger.info("\n" + "="*60)
logger.info("STEP 3: Prepare Training Data")
logger.info("="*60)

# Target: next period return
data_with_features['target'] = data_with_features['price'].pct_change(5).shift(-5)
data_clean = data_with_features.dropna()

# Select features
feature_cols = [col for col in data_clean.columns if col not in ['timestamp', 'target', 'price']]
X = data_clean[feature_cols].values
y = data_clean['target'].values

# Split
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]
X_test = X[train_size+val_size:]
y_test = y[train_size+val_size:]

logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 4. Train models
logger.info("\n" + "="*60)
logger.info("STEP 4: Model Training")
logger.info("="*60)

# XGBoost
xgb_model = BoostingAlphaModel(model_type='xgboost')
xgb_model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols)
xgb_pred = xgb_model.predict(X_test)
xgb_r2 = xgb_model.model.score(X_test, y_test)
logger.info(f"XGBoost Test R²: {xgb_r2:.4f}")

# Feature importance
top_features = xgb_model.get_feature_importance(top_n=10)
logger.info(f"\nTop 10 Features:\n{top_features}")

logger.info("\nAlpha research completed!")
