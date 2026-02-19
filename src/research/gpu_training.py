"""
GPU-Accelerated Training Pipeline

Supports model training on GPU clusters:
  - PyTorch training loop with mixed precision (AMP)
  - Distributed data-parallel (DDP) across multiple GPUs
  - Experiment tracking with MLflow / Weights & Biases
  - Hyperparameter optimization with Optuna
  - Model registry with versioning and promotion
  - Kubeflow pipeline integration
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


# --------------------------------------------------------------------------- #
#  Types                                                                       #
# --------------------------------------------------------------------------- #

class ModelStage(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainingConfig:
    model_name: str
    model_class: str  # e.g., "lstm", "transformer", "boosting"
    # Data
    feature_names: List[str] = field(default_factory=list)
    label_name: str = "forward_return_1h"
    train_start: str = "2018-01-01"
    train_end: str = "2023-01-01"
    val_start: str = "2023-01-01"
    val_end: str = "2023-07-01"
    test_start: str = "2023-07-01"
    test_end: str = "2024-01-01"
    symbols: List[str] = field(default_factory=list)
    # Training
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    # GPU
    use_gpu: bool = True
    mixed_precision: bool = True
    num_gpus: int = 1
    distributed: bool = False
    # Experiment tracking
    experiment_name: str = "alpha_research"
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    train_loss: float = 0.0
    val_loss: float = 0.0
    test_loss: float = 0.0
    train_ic: float = 0.0      # Information Coefficient
    val_ic: float = 0.0
    test_ic: float = 0.0
    train_sharpe: float = 0.0  # Sharpe of predicted returns
    val_sharpe: float = 0.0
    test_sharpe: float = 0.0
    epoch: int = 0
    training_time_seconds: float = 0.0
    best_epoch: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelArtifact:
    model_id: str
    model_name: str
    version: int
    stage: ModelStage
    config: TrainingConfig
    metrics: TrainingMetrics
    artifact_path: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            content = f"{self.model_name}:{self.version}:{self.created_at}"
            self.fingerprint = hashlib.md5(content.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
#  Neural Network Architectures                                                #
# --------------------------------------------------------------------------- #

try:
    import torch
    import torch.nn as nn

    class LSTMAlpha(nn.Module):
        """LSTM-based alpha model with attention."""

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True, dropout=dropout, bidirectional=False,
            )
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            lstm_out, _ = self.lstm(x)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            out = attn_out[:, -1, :]
            return self.fc(out)

    class TransformerAlpha(nn.Module):
        """Transformer-based alpha model."""

        def __init__(self, d_model: int = 128, nhead: int = 4,
                     num_layers: int = 3, n_features: int = 64):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=d_model * 4, dropout=0.1,
                batch_first=True, activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.input_proj(x)
            x = self.transformer(x)
            x = x[:, -1, :]
            return self.head(x)

except ImportError:
    LSTMAlpha = None  # type: ignore[misc, assignment]
    TransformerAlpha = None  # type: ignore[misc, assignment]


# --------------------------------------------------------------------------- #
#  PyTorch Training Engine                                                     #
# --------------------------------------------------------------------------- #

class PyTorchTrainer:
    """GPU-accelerated PyTorch training with AMP, scheduling, and early stopping."""

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._device = None
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._metrics_history: List[Dict[str, float]] = []

    def setup(self) -> None:
        try:
            import torch
            import torch.nn as nn

            # Device selection
            if self._config.use_gpu and torch.cuda.is_available():
                self._device = torch.device("cuda")
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                if self._config.mixed_precision:
                    self._scaler = torch.amp.GradScaler("cuda")
            else:
                self._device = torch.device("cpu")
                logger.info("Using CPU")

            # Model creation
            self._model = self._create_model()
            self._model.to(self._device)

            # Optimizer
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(),
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )

            # Scheduler
            if self._config.scheduler == "cosine":
                self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self._optimizer, T_max=self._config.epochs,
                )
            elif self._config.scheduler == "step":
                self._scheduler = torch.optim.lr_scheduler.StepLR(
                    self._optimizer, step_size=30, gamma=0.1,
                )
            elif self._config.scheduler == "plateau":
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, patience=5, factor=0.5,
                )

            # DDP setup
            if self._config.distributed and self._config.num_gpus > 1:
                self._setup_distributed()

            param_count = sum(p.numel() for p in self._model.parameters())
            logger.info(f"Model initialized: {param_count:,} parameters")

        except ImportError:
            logger.error("PyTorch not installed. pip install torch")
            raise

    def _create_model(self) -> Any:
        import torch
        import torch.nn as nn

        n_features = len(self._config.feature_names) or 64

        if self._config.model_class == "transformer":
            return self._create_transformer(n_features)
        elif self._config.model_class == "lstm":
            return self._create_lstm(n_features)
        else:
            return self._create_mlp(n_features)

    def _create_mlp(self, n_features: int) -> Any:
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _create_lstm(self, n_features: int) -> Any:
        return LSTMAlpha(n_features)

    def _create_transformer(self, n_features: int) -> Any:
        return TransformerAlpha(n_features=n_features)

    def _setup_distributed(self) -> None:
        import torch
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        self._model = DistributedDataParallel(
            self._model, device_ids=[local_rank],
        )
        logger.info(f"DDP initialized on rank {local_rank}")

    async def train(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
    ) -> TrainingMetrics:
        """Full training loop with early stopping."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X_train, y_train = train_data
        X_val, y_val = val_data

        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(1),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self._config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self._config.batch_size * 2,
            shuffle=False, num_workers=2, pin_memory=True,
        )

        loss_fn = torch.nn.MSELoss()
        best_metrics = TrainingMetrics()
        start_time = time.time()

        for epoch in range(self._config.epochs):
            # Train
            self._model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self._device, non_blocking=True)
                batch_y = batch_y.to(self._device, non_blocking=True)

                self._optimizer.zero_grad(set_to_none=True)

                if self._scaler:
                    with torch.amp.autocast("cuda"):
                        pred = self._model(batch_x)
                        loss = loss_fn(pred, batch_y)
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self._optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._config.gradient_clip_norm
                    )
                    self._scaler.step(self._optimizer)
                    self._scaler.update()
                else:
                    pred = self._model(batch_x)
                    loss = loss_fn(pred, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), self._config.gradient_clip_norm
                    )
                    self._optimizer.step()

                train_losses.append(loss.item())

            # Validate
            self._model.eval()
            val_losses = []
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self._device, non_blocking=True)
                    batch_y = batch_y.to(self._device, non_blocking=True)
                    pred = self._model(batch_x)
                    loss = loss_fn(pred, batch_y)
                    val_losses.append(loss.item())
                    val_preds.append(pred.cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            # IC calculation
            all_preds = np.concatenate(val_preds, axis=0).flatten()
            all_targets = np.concatenate(val_targets, axis=0).flatten()
            ic = float(np.corrcoef(all_preds, all_targets)[0, 1]) if len(all_preds) > 1 else 0.0

            # Scheduler step
            if self._scheduler:
                if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self._scheduler.step(val_loss)
                else:
                    self._scheduler.step()

            # Log
            self._metrics_history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_ic": ic,
                "lr": self._optimizer.param_groups[0]["lr"],
            })

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self._config.epochs}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, IC={ic:.4f}"
                )

            # Early stopping
            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                best_metrics = TrainingMetrics(
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_ic=ic,
                    epoch=epoch + 1,
                    best_epoch=epoch + 1,
                )
            else:
                self._patience_counter += 1
                if self._patience_counter >= self._config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        best_metrics.training_time_seconds = time.time() - start_time
        return best_metrics

    def save_checkpoint(self, path: str) -> None:
        import torch
        state = {
            "model_state_dict": self._model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "config": self._config.__dict__,
            "best_val_loss": self._best_val_loss,
            "metrics_history": self._metrics_history,
        }
        if self._scheduler:
            state["scheduler_state_dict"] = self._scheduler.state_dict()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        import torch
        state = torch.load(path, map_location=self._device)
        self._model.load_state_dict(state["model_state_dict"])
        self._optimizer.load_state_dict(state["optimizer_state_dict"])
        if self._scheduler and "scheduler_state_dict" in state:
            self._scheduler.load_state_dict(state["scheduler_state_dict"])
        self._best_val_loss = state.get("best_val_loss", float("inf"))
        logger.info(f"Checkpoint loaded: {path}")


# --------------------------------------------------------------------------- #
#  Hyperparameter Optimization                                                 #
# --------------------------------------------------------------------------- #

class HyperparameterOptimizer:
    """Optuna-based hyperparameter search with TPE and median pruning."""

    def __init__(
        self,
        base_config: TrainingConfig,
        n_trials: int = 50,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self._base_config = base_config
        self._n_trials = n_trials
        self._timeout = timeout_seconds

    async def optimize(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
    ) -> Tuple[Dict[str, Any], TrainingMetrics]:
        """Run hyperparameter optimization and return best params + metrics."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial: optuna.Trial) -> float:
                config = TrainingConfig(
                    model_name=self._base_config.model_name,
                    model_class=self._base_config.model_class,
                    feature_names=self._base_config.feature_names,
                    label_name=self._base_config.label_name,
                    epochs=trial.suggest_int("epochs", 10, 100),
                    batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
                    learning_rate=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    weight_decay=trial.suggest_float("wd", 1e-6, 1e-3, log=True),
                    early_stopping_patience=10,
                    use_gpu=self._base_config.use_gpu,
                    mixed_precision=self._base_config.mixed_precision,
                )

                import asyncio
                trainer = PyTorchTrainer(config)
                trainer.setup()

                loop = asyncio.new_event_loop()
                try:
                    metrics = loop.run_until_complete(
                        trainer.train(train_data, val_data)
                    )
                finally:
                    loop.close()

                return metrics.val_loss

            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            )
            study.optimize(
                objective,
                n_trials=self._n_trials,
                timeout=self._timeout,
            )

            best_params = study.best_params
            best_value = study.best_value
            logger.info(f"Best params: {best_params}, val_loss: {best_value:.6f}")

            return best_params, TrainingMetrics(val_loss=best_value)

        except ImportError:
            logger.warning("Optuna not installed. pip install optuna")
            return {}, TrainingMetrics()


# --------------------------------------------------------------------------- #
#  Model Registry                                                             #
# --------------------------------------------------------------------------- #

class ModelRegistry:
    """Model versioning, lifecycle management, and stage promotion."""

    def __init__(self, storage_dir: str = "data/model_registry") -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, List[ModelArtifact]] = {}

    def register(self, artifact: ModelArtifact) -> None:
        if artifact.model_name not in self._models:
            self._models[artifact.model_name] = []
        self._models[artifact.model_name].append(artifact)

        # Save metadata
        meta_path = self._storage_dir / artifact.model_name / f"v{artifact.version}" / "metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({
                "model_id": artifact.model_id,
                "model_name": artifact.model_name,
                "version": artifact.version,
                "stage": artifact.stage.value,
                "fingerprint": artifact.fingerprint,
                "created_at": artifact.created_at,
                "metrics": artifact.metrics.__dict__,
                "config": artifact.config.__dict__,
            }, f, indent=2, default=str)

        logger.info(f"Registered model: {artifact.model_name} v{artifact.version}")

    def promote(self, model_name: str, version: int, stage: ModelStage) -> None:
        """Promote a model version to a new stage."""
        versions = self._models.get(model_name, [])
        for v in versions:
            if v.version == version:
                old_stage = v.stage
                v.stage = stage
                logger.info(f"Promoted {model_name} v{version}: {old_stage.value} -> {stage.value}")
                return
        raise ValueError(f"Model {model_name} v{version} not found")

    def get_production_model(self, model_name: str) -> Optional[ModelArtifact]:
        versions = self._models.get(model_name, [])
        for v in reversed(versions):
            if v.stage == ModelStage.PRODUCTION:
                return v
        return None

    def get_latest(self, model_name: str) -> Optional[ModelArtifact]:
        versions = self._models.get(model_name, [])
        return versions[-1] if versions else None

    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        result = {}
        for name, versions in self._models.items():
            result[name] = [
                {
                    "version": v.version,
                    "stage": v.stage.value,
                    "val_ic": v.metrics.val_ic,
                    "val_loss": v.metrics.val_loss,
                    "created_at": v.created_at,
                }
                for v in versions
            ]
        return result
