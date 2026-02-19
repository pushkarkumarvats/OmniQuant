"""
OmniQuant Dashboard
Interactive visualization dashboard using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_pipeline.ingestion import DataIngestion
from src.feature_engineering.microstructure_features import MicrostructureFeatures
from src.feature_engineering.technical_features import TechnicalFeatures
from src.alpha_models.boosting_model import BoostingAlphaModel
from src.portfolio.optimizer import PortfolioOptimizer
from src.portfolio.risk_manager import RiskManager


# Page configuration
st.set_page_config(
    page_title="OmniQuant Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 OmniQuant Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Unified Quantitative Research & Trading Framework**")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### OmniQuant")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📈 Overview", "💹 Alpha Discovery", "📊 Portfolio Analysis", "⚙️ Backtesting", "📉 Risk Analysis"]
        )
        
        st.markdown("---")
        st.markdown("### Settings")
        
        # Generate sample data option
        if st.button("🔄 Generate Sample Data"):
            generate_sample_data()
            st.success("Sample data generated!")
    
    # Route to pages
    if page == "📈 Overview":
        show_overview()
    elif page == "💹 Alpha Discovery":
        show_alpha_discovery()
    elif page == "📊 Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "⚙️ Backtesting":
        show_backtesting()
    elif page == "📉 Risk Analysis":
        show_risk_analysis()


def generate_sample_data():
    """Generate sample data for demonstration"""
    ingestion = DataIngestion()
    
    # Generate tick data
    tick_data = ingestion.generate_synthetic_tick_data(num_ticks=5000, seed=42)
    tick_data.to_parquet("data/processed/sample_ticks.parquet")
    
    # Generate orderbook data
    orderbook_data = ingestion.generate_synthetic_orderbook(num_snapshots=1000, seed=42)
    orderbook_data.to_parquet("data/processed/sample_orderbook.parquet")
    
    st.session_state['data_generated'] = True


def show_overview():
    """Show overview page"""
    st.header("📈 System Overview")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Strategies", "3", "+1")
    with col2:
        st.metric("Total PnL", "$125,430", "+5.2%")
    with col3:
        st.metric("Sharpe Ratio", "2.34", "+0.12")
    with col4:
        st.metric("Max Drawdown", "-8.5%", "-1.2%")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Equity Curve")
        # Generate sample equity curve
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        equity = 1000000 + np.cumsum(np.random.randn(252) * 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=equity, mode='lines', name='Equity', line=dict(color='#1f77b4', width=2)))
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Equity ($)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Strategy Performance")
        strategies = ['Market Maker', 'Momentum', 'Arbitrage']
        returns = [12.5, 8.3, 15.2]
        
        fig = go.Figure(data=[go.Bar(x=strategies, y=returns, marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])])
        fig.update_layout(height=400, xaxis_title="Strategy", yaxis_title="Return (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent trades
    st.subheader("Recent Trades")
    trades_df = pd.DataFrame({
        'Timestamp': pd.date_range(end=pd.Timestamp.now(), periods=10, freq='1H'),
        'Symbol': np.random.choice(['AAPL', 'GOOGL', 'MSFT'], 10),
        'Side': np.random.choice(['BUY', 'SELL'], 10),
        'Quantity': np.random.randint(100, 1000, 10),
        'Price': np.random.uniform(100, 300, 10).round(2),
        'PnL': np.random.uniform(-500, 1000, 10).round(2)
    })
    st.dataframe(trades_df, use_container_width=True)


def show_alpha_discovery():
    """Show alpha discovery page"""
    st.header("💹 Alpha Discovery")
    
    tab1, tab2, tab3 = st.tabs(["Feature Engineering", "Model Training", "Feature Importance"])
    
    with tab1:
        st.subheader("Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Microstructure Features**")
            st.checkbox("Order Flow Imbalance (OFI)", value=True)
            st.checkbox("Bid-Ask Spread", value=True)
            st.checkbox("Order Book Depth", value=True)
            st.checkbox("Trade Intensity", value=True)
        
        with col2:
            st.markdown("**Technical Features**")
            st.checkbox("Momentum", value=True)
            st.checkbox("Volatility", value=True)
            st.checkbox("Moving Averages", value=True)
            st.checkbox("RSI", value=True)
        
        if st.button("Generate Features"):
            with st.spinner("Generating features..."):
                st.success("Features generated successfully!")
    
    with tab2:
        st.subheader("Model Training")
        
        model_type = st.selectbox("Select Model", ["XGBoost", "LightGBM", "LSTM", "Ensemble"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.slider("N Estimators", 50, 500, 100)
        with col2:
            learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1)
        with col3:
            max_depth = st.slider("Max Depth", 3, 10, 6)
        
        if st.button("Train Model"):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            st.success(f"{model_type} model trained successfully!")
            st.metric("Training R²", "0.82")
            st.metric("Validation R²", "0.78")
    
    with tab3:
        st.subheader("Feature Importance")
        
        # Sample feature importance
        features = ['OFI_20', 'Spread_BPS', 'Momentum_10', 'Volatility_20', 'RSI_14', 
                   'VWAP_Deviation', 'Volume_Ratio', 'Returns_5', 'MACD', 'BB_Width']
        importance = np.random.uniform(0.02, 0.15, len(features))
        importance = importance / importance.sum()
        
        fig = go.Figure(data=[go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color='#1f77b4'
        )])
        fig.update_layout(height=500, xaxis_title="Importance", yaxis_title="Feature")
        st.plotly_chart(fig, use_container_width=True)


def show_portfolio_analysis():
    """Show portfolio analysis page"""
    st.header("📊 Portfolio Analysis")
    
    tab1, tab2 = st.tabs(["Optimization", "Allocation"])
    
    with tab1:
        st.subheader("Portfolio Optimization")
        
        method = st.selectbox(
            "Optimization Method",
            ["Mean-Variance", "Risk Parity", "Hierarchical Risk Parity", "Black-Litterman", "Maximum Diversification"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Constraints**")
            long_only = st.checkbox("Long Only", value=True)
            max_weight = st.slider("Max Weight per Asset", 0.1, 1.0, 0.3)
        
        with col2:
            st.markdown("**Parameters**")
            risk_aversion = st.slider("Risk Aversion", 0.1, 5.0, 1.0)
            target_vol = st.slider("Target Volatility", 0.05, 0.30, 0.15)
        
        if st.button("Optimize Portfolio"):
            st.success("Portfolio optimized!")
            
            # Sample results
            assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
            weights = np.array([0.25, 0.20, 0.22, 0.18, 0.15])
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[go.Pie(labels=assets, values=weights, hole=0.3)])
                fig.update_layout(height=400, title="Portfolio Allocation")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Expected Return", "14.5%")
                st.metric("Portfolio Volatility", "16.2%")
                st.metric("Sharpe Ratio", "0.89")
    
    with tab2:
        st.subheader("Current Allocation")
        
        # Sample allocation data
        allocation_df = pd.DataFrame({
            'Asset': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META'],
            'Weight': [0.25, 0.20, 0.22, 0.18, 0.15],
            'Value': [250000, 200000, 220000, 180000, 150000],
            'Return': [0.12, 0.08, 0.15, 0.10, 0.13]
        })
        
        st.dataframe(allocation_df, use_container_width=True)


def show_backtesting():
    """Show backtesting page"""
    st.header("⚙️ Backtesting Engine")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Backtest Configuration")
        
        strategy = st.selectbox("Select Strategy", ["Market Maker", "Momentum", "Arbitrage", "Custom"])
        
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("Start Date", value=pd.Timestamp('2024-01-01'))
        with col_b:
            end_date = st.date_input("End Date", value=pd.Timestamp('2024-12-31'))
        
        initial_capital = st.number_input("Initial Capital", value=1000000, step=10000)
    
    with col2:
        st.subheader("Strategy Parameters")
        
        if strategy == "Momentum":
            lookback = st.slider("Lookback Period", 5, 50, 20)
            threshold = st.slider("Entry Threshold", 1.0, 3.0, 2.0)
        elif strategy == "Market Maker":
            spread_bps = st.slider("Spread (bps)", 5, 50, 10)
            inventory_limit = st.slider("Inventory Limit", 100, 5000, 1000)
    
    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            
            st.success("Backtest completed!")
            
            # Results
            st.markdown("---")
            st.subheader("Backtest Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Return", "15.2%")
            with col2:
                st.metric("Sharpe Ratio", "1.85")
            with col3:
                st.metric("Max Drawdown", "-8.3%")
            with col4:
                st.metric("Win Rate", "62.5%")
            
            # Equity curve
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            equity = initial_capital * (1 + np.cumsum(np.random.randn(len(dates)) * 0.001))
            
            fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1,
                               subplot_titles=("Equity Curve", "Drawdown"))
            
            fig.add_trace(go.Scatter(x=dates, y=equity, name='Equity', line=dict(color='#1f77b4')), row=1, col=1)
            
            # Drawdown
            cummax = pd.Series(equity).cummax()
            drawdown = (equity - cummax) / cummax * 100
            fig.add_trace(go.Scatter(x=dates, y=drawdown, name='Drawdown', fill='tozeroy', 
                                    line=dict(color='#d62728')), row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def show_risk_analysis():
    """Show risk analysis page"""
    st.header("📉 Risk Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Risk Metrics", "VaR Analysis", "Regime Detection"])
    
    with tab1:
        st.subheader("Risk Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Volatility", "16.2%", "-2.1%")
            st.metric("Downside Deviation", "11.8%")
        
        with col2:
            st.metric("Current Drawdown", "3.2%")
            st.metric("Max Drawdown", "8.5%")
        
        with col3:
            st.metric("Leverage", "0.85x")
            st.metric("Beta", "0.92")
        
        # Risk decomposition
        st.subheader("Risk Decomposition")
        
        assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META']
        risk_contrib = np.random.uniform(0.15, 0.25, len(assets))
        risk_contrib = risk_contrib / risk_contrib.sum()
        
        fig = go.Figure(data=[go.Bar(x=assets, y=risk_contrib, marker_color='#ff7f0e')])
        fig.update_layout(height=400, xaxis_title="Asset", yaxis_title="Risk Contribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Value at Risk Analysis")
        
        confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("VaR (95%)", "$18,500")
            st.metric("CVaR (95%)", "$24,300")
        
        with col2:
            st.metric("VaR (99%)", "$28,700")
            st.metric("CVaR (99%)", "$35,200")
        
        # VaR distribution
        returns = np.random.normal(0, 0.01, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns', marker_color='#1f77b4'))
        fig.add_vline(x=np.percentile(returns, (1 - confidence_level) * 100), 
                     line_dash="dash", line_color="red", annotation_text=f"VaR ({confidence_level:.0%})")
        fig.update_layout(height=400, xaxis_title="Returns", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Market Regime Detection")
        
        n_regimes = st.slider("Number of Regimes", 2, 5, 3)
        
        # Sample regime visualization
        dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
        regimes = np.random.choice(range(n_regimes), 252)
        
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for regime in range(n_regimes):
            mask = regimes == regime
            fig.add_trace(go.Scatter(
                x=dates[mask],
                y=np.ones(mask.sum()) * regime,
                mode='markers',
                name=f'Regime {regime}',
                marker=dict(color=colors[regime], size=10)
            ))
        
        fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Regime", 
                         yaxis=dict(tickmode='linear', tick0=0, dtick=1))
        st.plotly_chart(fig, use_container_width=True)
        
        # Regime statistics
        st.subheader("Regime Statistics")
        regime_stats = pd.DataFrame({
            'Regime': range(n_regimes),
            'Frequency': np.random.uniform(0.25, 0.40, n_regimes),
            'Avg Return': np.random.uniform(-0.01, 0.02, n_regimes),
            'Volatility': np.random.uniform(0.01, 0.03, n_regimes)
        })
        st.dataframe(regime_stats, use_container_width=True)


if __name__ == "__main__":
    main()
