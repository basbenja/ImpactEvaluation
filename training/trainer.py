import torch
import torch.nn as nn

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

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
    ) -> None:
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_step(train_loader)
            val_loss   = self._validation_step(val_loader)

            width = len(str(n_epochs))
            print(
                f"Epoch {epoch:{width}}/{n_epochs} — "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f}"
            )


    def _train_step(self, loader: DataLoader) -> float:
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
