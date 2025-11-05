"""
Parametric DNN training for DarkBottomLine framework.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import awkward as ak
import yaml
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt


class ParametricDNN(nn.Module):
    """
    Parametric Deep Neural Network for signal-background classification.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize parametric DNN.

        Args:
            config: Model configuration dictionary
        """
        super(ParametricDNN, self).__init__()

        self.config = config
        self.input_features = config.get("input_features", 20)
        self.hidden_layers = config.get("hidden_layers", [128, 64, 32])
        self.activation = config.get("activation", "relu")
        self.dropout = config.get("dropout", 0.2)
        self.parametric_input = config.get("parametric_input", True)
        self.output_activation = config.get("output_activation", "sigmoid")

        # Build network layers
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input layer
        input_size = self.input_features
        if self.parametric_input:
            input_size += 1  # Add mass parameter

        # Hidden layers
        for hidden_size in self.hidden_layers:
            self.layers.append(nn.Linear(input_size, hidden_size))
            self.dropouts.append(nn.Dropout(self.dropout))
            input_size = hidden_size

        # Output layer
        self.output_layer = nn.Linear(input_size, 1)

        # Activation functions
        self.activation_fn = self._get_activation(self.activation)
        self.output_activation_fn = self._get_activation(self.output_activation)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "gelu":
            return nn.GELU()
        else:
            return nn.ReLU()

    def forward(self, x: torch.Tensor, mass: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input features tensor
            mass: Mass parameter tensor (optional)

        Returns:
            Output tensor
        """
        # Add mass parameter if provided
        if self.parametric_input and mass is not None:
            x = torch.cat([x, mass.unsqueeze(-1)], dim=-1)

        # Forward through hidden layers
        for layer, dropout in zip(self.layers, self.dropouts):
            x = layer(x)
            x = self.activation_fn(x)
            x = dropout(x)

        # Output layer
        x = self.output_layer(x)
        x = self.output_activation_fn(x)

        return x


class DNNTrainer:
    """
    DNN trainer for parametric signal-background classification.
    """

    def __init__(self, config_path: str):
        """
        Initialize DNN trainer.

        Args:
            config_path: Path to DNN configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get("model", {})
        self.training_config = self.config.get("training", {})
        self.preprocessing_config = self.config.get("preprocessing", {})
        self.features_config = self.config.get("features", {})
        self.data_config = self.config.get("data", {})
        self.output_config = self.config.get("output", {})
        self.evaluation_config = self.config.get("evaluation", {})

        # Initialize model
        self.model = ParametricDNN(self.model_config)

        # Initialize preprocessing
        self.scaler = self._get_scaler()

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_auc": [],
            "val_auc": []
        }

        # Device configuration
        self.device = self._get_device()
        self.model.to(self.device)

        logging.info(f"Initialized DNN trainer with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _get_scaler(self):
        """Get feature scaler."""
        scaling_type = self.preprocessing_config.get("feature_scaling", "standard")

        if scaling_type == "standard":
            return StandardScaler()
        elif scaling_type == "minmax":
            return MinMaxScaler()
        elif scaling_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()

    def _get_device(self):
        """Get computation device."""
        device_config = self.config.get("hardware", {}).get("device", "auto")

        if device_config == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(device_config)

    def load_data(self, signal_files: List[str], background_files: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load training data from files.

        Args:
            signal_files: List of signal file paths
            background_files: List of background file paths

        Returns:
            Tuple of (features, labels, masses)
        """
        logging.info("Loading training data...")

        # Load signal data
        signal_features = []
        signal_labels = []
        signal_masses = []

        for signal_file in signal_files:
            # Load signal events (placeholder implementation)
            # In real implementation, this would load from ROOT files
            features, masses = self._load_sample_file(signal_file, is_signal=True)
            signal_features.append(features)
            signal_labels.append(np.ones(len(features)))
            signal_masses.append(masses)

        # Load background data
        background_features = []
        background_labels = []
        background_masses = []

        for background_file in background_files:
            # Load background events (placeholder implementation)
            features, masses = self._load_sample_file(background_file, is_signal=False)
            background_features.append(features)
            background_labels.append(np.zeros(len(features)))
            background_masses.append(masses)

        # Combine data
        all_features = np.vstack(signal_features + background_features)
        all_labels = np.hstack(signal_labels + background_labels)
        all_masses = np.hstack(signal_masses + background_masses)

        logging.info(f"Loaded {len(all_features)} events ({np.sum(all_labels)} signal, {np.sum(1-all_labels)} background)")

        return all_features, all_labels, all_masses

    def _load_sample_file(self, file_path: str, is_signal: bool) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sample file (placeholder implementation).

        Args:
            file_path: Path to sample file
            is_signal: Whether this is a signal sample

        Returns:
            Tuple of (features, masses)
        """
        # Placeholder implementation
        # In real implementation, this would load from ROOT files and extract features

        n_events = 10000  # Placeholder
        n_features = self.model_config.get("input_features", 20)

        # Generate random features
        features = np.random.randn(n_events, n_features)

        # Generate masses
        if is_signal:
            # Signal masses from configuration
            signal_masses = [1000, 1200, 1400, 1600, 1800, 2000]
            masses = np.random.choice(signal_masses, n_events)
        else:
            # Background has no mass parameter
            masses = np.zeros(n_events)

        return features, masses

    def preprocess(self, features: np.ndarray, labels: np.ndarray, masses: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Preprocess data for training.

        Args:
            features: Feature array
            labels: Label array
            masses: Mass parameter array

        Returns:
            Tuple of (train_features, train_labels, val_features, val_labels)
        """
        logging.info("Preprocessing data...")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train/validation split
        split_ratio = self.preprocessing_config.get("train_validation_split", 0.8)
        n_train = int(len(features_scaled) * split_ratio)

        # Shuffle data
        if self.preprocessing_config.get("shuffle_data", True):
            indices = np.random.permutation(len(features_scaled))
            features_scaled = features_scaled[indices]
            labels = labels[indices]
            masses = masses[indices]

        # Split data
        train_features = features_scaled[:n_train]
        train_labels = labels[:n_train]
        train_masses = masses[:n_train]

        val_features = features_scaled[n_train:]
        val_labels = labels[n_train:]
        val_masses = masses[n_train:]

        # Convert to tensors
        train_features = torch.FloatTensor(train_features)
        train_labels = torch.FloatTensor(train_labels)
        train_masses = torch.FloatTensor(train_masses)

        val_features = torch.FloatTensor(val_features)
        val_labels = torch.FloatTensor(val_labels)
        val_masses = torch.FloatTensor(val_masses)

        logging.info(f"Preprocessed data: {len(train_features)} train, {len(val_features)} validation")

        return train_features, train_labels, train_masses, val_features, val_labels, val_masses

    def train(self, train_features: torch.Tensor, train_labels: torch.Tensor, train_masses: torch.Tensor,
              val_features: torch.Tensor, val_labels: torch.Tensor, val_masses: torch.Tensor) -> Dict[str, Any]:
        """
        Train the DNN model.

        Args:
            train_features: Training features
            train_labels: Training labels
            train_masses: Training masses
            val_features: Validation features
            val_labels: Validation labels
            val_masses: Validation masses

        Returns:
            Training results
        """
        logging.info("Starting DNN training...")

        # Create data loaders
        batch_size = self.training_config.get("batch_size", 1024)
        train_dataset = TensorDataset(train_features, train_labels, train_masses)
        val_dataset = TensorDataset(val_features, val_labels, val_masses)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize optimizer
        learning_rate = self.training_config.get("learning_rate", 0.001)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize loss function
        criterion = nn.BCELoss()

        # Training parameters
        epochs = self.training_config.get("epochs", 100)
        early_stopping_patience = self.training_config.get("early_stopping", {}).get("patience", 10)
        min_delta = self.training_config.get("early_stopping", {}).get("min_delta", 0.001)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            train_auc = 0.0

            for batch_features, batch_labels, batch_masses in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                batch_masses = batch_masses.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_features, batch_masses).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Calculate accuracy
                predictions = (outputs > 0.5).float()
                train_acc += (predictions == batch_labels).float().mean().item()

                # Calculate AUC
                if len(torch.unique(batch_labels)) > 1:
                    train_auc += roc_auc_score(batch_labels.cpu().numpy(), outputs.detach().cpu().numpy())

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_auc = 0.0

            with torch.no_grad():
                for batch_features, batch_labels, batch_masses in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    batch_masses = batch_masses.to(self.device)

                    outputs = self.model(batch_features, batch_masses).squeeze()
                    loss = criterion(outputs, batch_labels)

                    val_loss += loss.item()

                    # Calculate accuracy
                    predictions = (outputs > 0.5).float()
                    val_acc += (predictions == batch_labels).float().mean().item()

                    # Calculate AUC
                    if len(torch.unique(batch_labels)) > 1:
                        val_auc += roc_auc_score(batch_labels.cpu().numpy(), outputs.cpu().numpy())

            # Average metrics
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            train_auc /= len(train_loader)

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_auc /= len(val_loader)

            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["train_auc"].append(train_auc)
            self.history["val_auc"].append(val_auc)

            # Log progress
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                           f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
                           f"train_auc={train_auc:.4f}, val_auc={val_auc:.4f}")

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model(self.output_config.get("model_path", "model.pt"))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

        logging.info("Training completed!")

        return {
            "best_val_loss": best_val_loss,
            "final_epoch": epoch,
            "history": self.history
        }

    def evaluate(self, features: torch.Tensor, labels: torch.Tensor, masses: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            features: Feature tensor
            labels: Label tensor
            masses: Mass tensor

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(features.to(self.device), masses.to(self.device)).squeeze()
            predictions = (outputs > 0.5).float()

            # Calculate metrics
            accuracy = (predictions == labels.to(self.device)).float().mean().item()

            if len(torch.unique(labels)) > 1:
                auc = roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy())
            else:
                auc = 0.0

            # Calculate precision and recall
            true_positives = ((predictions == 1) & (labels.to(self.device) == 1)).sum().item()
            false_positives = ((predictions == 1) & (labels.to(self.device) == 0)).sum().item()
            false_negatives = ((predictions == 0) & (labels.to(self.device) == 1)).sum().item()

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def save_model(self, model_path: str):
        """
        Save the trained model.

        Args:
            model_path: Path to save the model
        """
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'scaler': self.scaler,
            'history': self.history
        }, model_path)

        logging.info(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """
        Load a trained model.

        Args:
            model_path: Path to the model file
        """
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.history = checkpoint.get('history', {})

        logging.info(f"Model loaded from {model_path}")

    def predict(self, features: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            features: Feature array
            masses: Mass parameter array

        Returns:
            Predictions array
        """
        self.model.eval()

        # Preprocess features
        features_scaled = self.scaler.transform(features)

        # Convert to tensors
        features_tensor = torch.FloatTensor(features_scaled)
        masses_tensor = torch.FloatTensor(masses)

        # Make predictions
        with torch.no_grad():
            outputs = self.model(features_tensor.to(self.device), masses_tensor.to(self.device))
            predictions = outputs.cpu().numpy()

        return predictions

    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history.

        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss plot
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.history["val_loss"], label="Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy plot
        axes[0, 1].plot(self.history["train_acc"], label="Train Accuracy")
        axes[0, 1].plot(self.history["val_acc"], label="Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # AUC plot
        axes[1, 0].plot(self.history["train_auc"], label="Train AUC")
        axes[1, 0].plot(self.history["val_auc"], label="Validation AUC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].set_title("Training and Validation AUC")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning curve
        axes[1, 1].plot(self.history["train_loss"], label="Train Loss")
        axes[1, 1].plot(self.history["val_loss"], label="Validation Loss")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Learning Curve")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training history plot saved to {save_path}")

        plt.show()
