"""
Transformer Model for Time Series Forecasting
State-of-the-art attention-based model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from loguru import logger


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1)]


class TimeSeriesTransformer(nn.Module):
    """Encoder-decoder transformer for time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        logger.info(f"Transformer initialized: d_model={d_model}, nhead={nhead}, layers={num_encoder_layers}")
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project input
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Apply transformer
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to output dimension
        output = self.output_projection(output)
        
        return output


class TransformerAlphaModel:
    """Wrapper around TimeSeriesTransformer for alpha prediction."""
    
    def __init__(
        self,
        input_dim: int,
        config: Optional[dict] = None
    ):
        self.config = config or {}
        self.input_dim = input_dim
        
        # Model hyperparameters
        self.d_model = self.config.get('d_model', 128)
        self.nhead = self.config.get('nhead', 8)
        self.num_layers = self.config.get('num_layers', 3)
        self.dropout = self.config.get('dropout', 0.1)
        
        # Training parameters
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.num_epochs = self.config.get('num_epochs', 100)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        logger.info(f"Transformer alpha model initialized on {self.device}")
    
    def prepare_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 50,
        pred_len: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build encoder/decoder/target tensors from rolling windows over X and y."""
        n_samples = len(X) - seq_len - pred_len + 1
        
        encoder_inputs = []
        decoder_inputs = []
        targets = []
        
        for i in range(n_samples):
            # Encoder input: historical features
            enc_input = X[i:i+seq_len]
            
            # Decoder input: shifted target (teacher forcing)
            dec_input = np.zeros((pred_len, X.shape[1]))
            dec_input[0] = X[i+seq_len-1]  # Last encoder input as first decoder input
            
            # Target: future returns
            target = y[i+seq_len:i+seq_len+pred_len]
            
            encoder_inputs.append(enc_input)
            decoder_inputs.append(dec_input)
            targets.append(target)
        
        return (
            torch.FloatTensor(np.array(encoder_inputs)),
            torch.FloatTensor(np.array(decoder_inputs)),
            torch.FloatTensor(np.array(targets))
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        seq_len: int = 50
    ):
        """Train the transformer with gradient clipping and periodic validation logging."""
        # Prepare data
        enc_train, dec_train, tgt_train = self.prepare_sequences(X_train, y_train, seq_len)
        
        train_dataset = torch.utils.data.TensorDataset(enc_train, dec_train, tgt_train)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            
            for batch_idx, (enc_input, dec_input, target) in enumerate(train_loader):
                enc_input = enc_input.to(self.device)
                dec_input = dec_input.to(self.device)
                target = target.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(enc_input, dec_input)
                
                # Calculate loss
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            if X_val is not None and epoch % 10 == 0:
                val_loss = self.evaluate(X_val, y_val, seq_len)
                logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.6f}, Val Loss={val_loss:.6f}")
            else:
                logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.6f}")
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 50
    ) -> float:
        self.model.eval()
        
        enc_input, dec_input, target = self.prepare_sequences(X, y, seq_len)
        
        dataset = torch.utils.data.TensorDataset(enc_input, dec_input, target)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        
        total_loss = 0
        with torch.no_grad():
            for enc, dec, tgt in loader:
                enc = enc.to(self.device)
                dec = dec.to(self.device)
                tgt = tgt.to(self.device)
                
                output = self.model(enc, dec)
                loss = self.criterion(output, tgt)
                total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def predict(
        self,
        X: np.ndarray,
        seq_len: int = 50
    ) -> np.ndarray:
        self.model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(len(X) - seq_len + 1):
                enc_input = torch.FloatTensor(X[i:i+seq_len]).unsqueeze(0).to(self.device)
                dec_input = torch.FloatTensor(X[i+seq_len-1:i+seq_len]).unsqueeze(0).to(self.device)
                
                output = self.model(enc_input, dec_input)
                predictions.append(output.cpu().numpy()[0, 0, 0])
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    input_dim = 50
    model = TransformerAlphaModel(input_dim=input_dim)
    
    # Generate sample data
    X_train = np.random.randn(1000, input_dim)
    y_train = np.random.randn(1000, 1)
    
    # Train
    model.train(X_train, y_train)
    
    # Predict
    predictions = model.predict(X_train[:100])
    print(f"Predictions shape: {predictions.shape}")
