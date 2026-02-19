"""
LSTM Alpha Model
Deep learning model for time series prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from loguru import logger
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMNetwork(nn.Module):
    """LSTM neural network"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.relu(self.fc1(last_output))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        
        return out.squeeze()


class LSTMAlphaModel:
    """LSTM-based alpha prediction model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Model parameters
        self.sequence_length = self.config.get('sequence_length', 50)
        self.hidden_size = self.config.get('hidden_size', 128)
        self.num_layers = self.config.get('num_layers', 2)
        self.dropout = self.config.get('dropout', 0.2)
        self.bidirectional = self.config.get('bidirectional', False)
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 32)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.num_epochs = self.config.get('num_epochs', 100)
        self.patience = self.config.get('patience', 10)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sequence_length: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slide a window over X/y to build (sequence, target) pairs for the LSTM."""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping and LR scheduling.
        Returns a history dict of per-epoch losses.
        """
        logger.info("Starting LSTM model training")
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train_scaled)
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val_scaled)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None:
            val_dataset = TimeSeriesDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val is not None:
                self.model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                history['val_loss'].append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {avg_train_loss:.6f}")
        
        logger.info("Training completed")
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale features
        X_scaled = self.scaler_X.transform(X)
        
        # prepare_sequences needs a y array for alignment; targets are
        # unused at inference time, so we pass a zero placeholder.
        placeholder_y = np.zeros(len(X_scaled))
        X_seq, _ = self.prepare_sequences(X_scaled, placeholder_y)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform
        predictions = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        
        return predictions
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("No model to save")
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'config': self.config
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.scaler_X = checkpoint['scaler_X']
        self.scaler_y = checkpoint['scaler_y']
        
        # Rebuild model
        input_size = self.scaler_X.n_features_in_
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Target: simple linear combination with noise
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    
    # Train model
    model = LSTMAlphaModel(config={
        'sequence_length': 20,
        'hidden_size': 64,
        'num_layers': 2,
        'num_epochs': 50
    })
    
    history = model.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Calculate MSE
    mse = np.mean((predictions - y_test[model.sequence_length:]) ** 2)
    logger.info(f"Test MSE: {mse:.6f}")
