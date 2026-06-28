import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device = 'cpu',
    ):
        self.model     = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device    = device

    def _train_step(self, loader: DataLoader) -> float:
        """
        Perform a single training step (one epoch) over the provided DataLoader.
        Returns the average loss over the epoch (across all batches).
        """
        self.model.train()
        total_loss = 0.0

        for *X, y in loader:
            X = [x.to(self.device) for x in X]
            y = y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(*X)
            loss   = self.criterion(logits.squeeze(1), y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        # average loss per batch
        return total_loss / len(loader)

    def _validation_step(self, loader: DataLoader) -> float:
        """
        Perform a single validation step (one epoch) over the provided DataLoader.
        Returns the average loss over the epoch (across all batches).
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for *X, y in loader:
                X = [x.to(self.device) for x in X]
                y = y.to(self.device)

                logits = self.model(*X)
                loss   = self.criterion(logits.squeeze(1), y)
                total_loss += loss.item()

        # average loss per batch
        return total_loss / len(loader)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> None:
        """
        Train the model for the specified number of epochs.
        Performs one training step using the tran_loader and one validation step
        using the val_loader for each epoch.
        Prints the training and validation loss for each epoch.
        """
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_step(train_loader)
            val_loss   = self._validation_step(val_loader)

            width = len(str(n_epochs))
            print(
                f"Epoch {epoch:{width}}/{n_epochs} — "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f}"
            )

    def _predict(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on the provided DataLoader.
        Returns a tuple of true labels and predicted PROBABILITIES (NOT labels).
        """
        self.model.eval()
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for *X, y in loader:
                X = [x.to(self.device) for x in X]
                logits = self.model(*X).squeeze(1)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(y)

        return torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy()

    def accuracy(self, loader: DataLoader, threshold: float = 0.5) -> float:
        y_true, y_prob = self._predict(loader)
        return float(((y_prob >= threshold) == y_true).mean())

    def f1(self, loader: DataLoader, threshold: float = 0.5) -> float:
        y_true, y_prob = self._predict(loader)
        return float(f1_score(y_true, (y_prob >= threshold).astype(int)))

    def precision(self, loader: DataLoader, threshold: float = 0.5) -> float:
        y_true, y_prob = self._predict(loader)
        return float(precision_score(y_true, (y_prob >= threshold).astype(int)))

    def recall(self, loader: DataLoader, threshold: float = 0.5) -> float:
        y_true, y_prob = self._predict(loader)
        return float(recall_score(y_true, (y_prob >= threshold).astype(int)))

    def roc_auc(self, loader: DataLoader) -> float:
        y_true, y_prob = self._predict(loader)
        return float(roc_auc_score(y_true, y_prob))

    def compute_metric(self, name: str, loader: DataLoader, threshold: float = 0.5) -> float:
        if name == 'roc_auc':
            return self.roc_auc(loader)
        return getattr(self, name)(loader, threshold=threshold)
