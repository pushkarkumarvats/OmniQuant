"""
Causal Features
Causal inference and feature interaction analysis
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from loguru import logger
from scipy import stats
import networkx as nx


class CausalFeatures:
    """Causal inference and feature interaction analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.causal_graph = None
        
    def granger_causality(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        max_lag: int = 5,
        significance: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """Test Granger causality of each feature on the target, returning p-values and optimal lags."""
        from statsmodels.tsa.stattools import grangercausalitytests
        
        results = {}
        
        for feature in feature_cols:
            if feature == target_col or feature not in df.columns:
                continue
            
            try:
                # Prepare data
                data = df[[target_col, feature]].dropna()
                
                if len(data) < max_lag + 10:
                    continue
                
                # Run Granger causality test
                gc_result = grangercausalitytests(data, max_lag, verbose=False)
                
                # Extract p-values
                p_values = []
                for lag in range(1, max_lag + 1):
                    p_value = gc_result[lag][0]['ssr_ftest'][1]
                    p_values.append(p_value)
                
                min_p_value = min(p_values)
                is_causal = min_p_value < significance
                
                results[feature] = {
                    'p_values': p_values,
                    'min_p_value': min_p_value,
                    'is_causal': is_causal,
                    'optimal_lag': p_values.index(min_p_value) + 1
                }
                
                if is_causal:
                    logger.debug(f"{feature} Granger-causes {target_col} (p={min_p_value:.4f}, lag={results[feature]['optimal_lag']})")
                    
            except Exception as e:
                logger.warning(f"Granger causality test failed for {feature}: {e}")
                continue
        
        return results
    
    def mutual_information(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        n_neighbors: int = 3
    ) -> pd.Series:
        """MI scores between features and target via sklearn's KNN estimator."""
        from sklearn.feature_selection import mutual_info_regression
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Calculate MI
        mi_scores = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=42)
        
        mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
        
        logger.info(f"Calculated mutual information for {len(feature_cols)} features")
        return mi_series
    
    def feature_interaction_strength(
        self,
        df: pd.DataFrame,
        feature1: str,
        feature2: str,
        target: str
    ) -> float:
        # Create interaction term
        interaction = df[feature1] * df[feature2]
        
        # Compare correlation with and without interaction
        corr_f1 = abs(df[feature1].corr(df[target]))
        corr_f2 = abs(df[feature2].corr(df[target]))
        corr_int = abs(interaction.corr(df[target]))
        
        # Interaction strength is how much interaction improves over individual features
        baseline = max(corr_f1, corr_f2)
        strength = max(0, corr_int - baseline)
        
        return strength
    
    def build_feature_graph(
        self,
        df: pd.DataFrame,
        features: List[str],
        method: str = 'correlation',
        threshold: float = 0.5
    ) -> nx.Graph:
        """Build a NetworkX graph with edges between features exceeding the similarity threshold."""
        G = nx.Graph()
        
        # Add nodes
        for feature in features:
            if feature in df.columns:
                G.add_node(feature)
        
        # Add edges based on relationships
        if method == 'correlation':
            corr_matrix = df[features].corr().abs()
            
            for i, feat1 in enumerate(features):
                for j, feat2 in enumerate(features):
                    if i < j and feat1 in df.columns and feat2 in df.columns:
                        corr = corr_matrix.loc[feat1, feat2]
                        if corr > threshold:
                            G.add_edge(feat1, feat2, weight=corr)
        
        logger.info(f"Built feature graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        self.causal_graph = G
        return G
    
    def find_confounders(
        self,
        df: pd.DataFrame,
        treatment: str,
        outcome: str,
        features: List[str],
        threshold: float = 0.3
    ) -> List[str]:
        """Find features that correlate with both treatment and outcome above threshold."""
        confounders = []
        
        for feature in features:
            if feature == treatment or feature == outcome or feature not in df.columns:
                continue
            
            # A confounder should correlate with both treatment and outcome
            corr_treatment = abs(df[feature].corr(df[treatment]))
            corr_outcome = abs(df[feature].corr(df[outcome]))
            
            if corr_treatment > threshold and corr_outcome > threshold:
                confounders.append(feature)
                logger.debug(f"Potential confounder: {feature} (treatment: {corr_treatment:.3f}, outcome: {corr_outcome:.3f})")
        
        return confounders
    
    def transfer_entropy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lag: int = 1,
        bins: int = 10
    ) -> float:
        """Transfer entropy TE(X->Y) via binned plug-in estimator."""
        # Discretize
        x_binned = pd.cut(x, bins=bins, labels=False, duplicates='drop')
        y_binned = pd.cut(y, bins=bins, labels=False, duplicates='drop')
        
        # Shift for lag
        y_past = y_binned[:-lag]
        y_future = y_binned[lag:]
        x_past = x_binned[:-lag]

        # Align lengths and drop NaNs
        mask = ~(np.isnan(y_past) | np.isnan(y_future) | np.isnan(x_past))
        y_past = y_past[mask].astype(int)
        y_future = y_future[mask].astype(int)
        x_past = x_past[mask].astype(int)

        n = len(y_past)
        if n < 10:
            return 0.0

        # Build joint and marginal count tables, then compute
        # TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
        # via KSG-style plug-in estimator with add-one smoothing.
        from collections import Counter

        joint_ypast_xpast_yfut = Counter(zip(y_past, x_past, y_future))
        joint_ypast_xpast       = Counter(zip(y_past, x_past))
        joint_ypast_yfut        = Counter(zip(y_past, y_future))
        marginal_ypast          = Counter(y_past)

        te = 0.0
        for (yp, xp, yf), count in joint_ypast_xpast_yfut.items():
            p_ypxpyf = count / n
            p_yf_given_ypxp = count / joint_ypast_xpast[(yp, xp)]
            p_yf_given_yp   = joint_ypast_yfut[(yp, yf)] / marginal_ypast[yp]
            if p_yf_given_ypxp > 0 and p_yf_given_yp > 0:
                te += p_ypxpyf * np.log2(p_yf_given_ypxp / p_yf_given_yp)

        return max(te, 0.0)
    
    def lag_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: List[str],
        max_lag: int = 10
    ) -> pd.DataFrame:
        """Correlation of each feature at each lag with the target."""
        results = []
        
        for feature in feature_cols:
            if feature not in df.columns or feature == target_col:
                continue
            
            for lag in range(1, max_lag + 1):
                # Create lagged feature
                lagged = df[feature].shift(lag)
                
                # Calculate correlation
                corr = abs(lagged.corr(df[target_col]))
                
                results.append({
                    'feature': feature,
                    'lag': lag,
                    'correlation': corr
                })
        
        results_df = pd.DataFrame(results)
        
        # Find optimal lag for each feature
        optimal_lags = results_df.groupby('feature')['correlation'].idxmax()
        
        logger.info(f"Analyzed lag importance for {len(feature_cols)} features up to lag {max_lag}")
        return results_df
    
    def create_lagged_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        lags: List[int],
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        df_lagged = df.copy()
        
        for feature in features:
            if feature not in df.columns:
                continue
            
            for lag in lags:
                lag_name = f"{feature}_lag{lag}"
                df_lagged[lag_name] = df[feature].shift(lag)
        
        logger.info(f"Created {len(features) * len(lags)} lagged features")
        return df_lagged
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]],
        operations: List[str] = ['multiply', 'add', 'divide', 'subtract']
    ) -> pd.DataFrame:
        """Generate pairwise feature combinations (multiply, add, subtract, divide)."""
        df_interactions = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 not in df.columns or feat2 not in df.columns:
                continue
            
            if 'multiply' in operations:
                df_interactions[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            
            if 'add' in operations:
                df_interactions[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
            
            if 'subtract' in operations:
                df_interactions[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
            
            if 'divide' in operations:
                df_interactions[f'{feat1}_div_{feat2}'] = df[feat1] / df[feat2].replace(0, np.nan)
        
        logger.info(f"Created interaction features for {len(feature_pairs)} pairs")
        return df_interactions
    
    def polynomial_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        from sklearn.preprocessing import PolynomialFeatures
        
        # Extract feature data
        X = df[features].fillna(0)
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Create feature names
        feature_names = poly.get_feature_names_out(features)
        
        # Create DataFrame
        df_poly = pd.DataFrame(X_poly, columns=feature_names, index=df.index)
        
        # Combine with original
        df_result = pd.concat([df, df_poly], axis=1)
        
        logger.info(f"Created polynomial features up to degree {degree}")
        return df_result


if __name__ == "__main__":
    # Example usage
    from src.data_pipeline.ingestion import DataIngestion
    from src.feature_engineering.technical_features import TechnicalFeatures
    
    # Generate synthetic data
    ingestion = DataIngestion()
    tick_data = ingestion.generate_synthetic_tick_data(num_ticks=5000, seed=42)
    
    # Convert to OHLCV
    df = tick_data.set_index('timestamp').resample('1min').agg({
        'price': 'last',
        'volume': 'sum',
        'bid': 'last',
        'ask': 'last'
    }).reset_index()
    
    # Add some features
    tech_features = TechnicalFeatures()
    df = tech_features.generate_all_features(df, price_col='price')
    
    # Causal analysis
    causal = CausalFeatures()
    
    # Mutual information
    mi_scores = causal.mutual_information(
        df, 
        target_col='return_1',
        feature_cols=['return_5', 'return_10', 'momentum_10', 'rsi_14']
    )
    print("Top features by MI:")
    print(mi_scores.head())
    
    # Create lagged features
    df_lagged = causal.create_lagged_features(
        df,
        features=['price', 'volume'],
        lags=[1, 2, 5]
    )
    
    logger.info("Causal features example completed")
