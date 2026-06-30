import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from training.trainer import Trainer
from connectors.base import BaseConnector


class CrossValidator:
    def __init__(
        self,
        connector: BaseConnector,
        dataset_factory: Dataset,
        model_factory,
        optimizer_factory,
        criterion: nn.Module,
        collate_fn,
        k: int = 5,
        n_epochs: int = 10,
        batch_size: int = 32,
        device: torch.device = None,
        scale: bool = False,
        metric: str = 'accuracy',
        threshold: float = 0.5,
    ):
        """
        connector: needed for building train records and scaling on each fold

        dataset_factory: a callable that takes a list of records and returns a
            Dataset. Needed to create the loaders for each fold.

        model_factory: a callable that returns a new instance of the model.
            Needed to create a new fresh model for each fold.

        optimizer_factory: a callable that takes a model and returns a new
            optimizer. Needed to create a new fresh optimizer for each fold.
            Remember the optimizer takes the model parameters as input, so it
            needs to be created after the model.

        criterion: the loss function to use for training. Can be shared across
            folds.

        collate_fn: the collate function to use for the DataLoader. Can be
            shared across folds. Needed because LSTM input sequences have
            different lengths (different firms have different pre-treatment
            windows).
        """
        self.connector         = connector
        self.dataset_factory   = dataset_factory
        self.model_factory     = model_factory
        self.optimizer_factory = optimizer_factory
        self.criterion         = criterion
        self.collate_fn        = collate_fn
        self.k                 = k
        self.n_epochs          = n_epochs
        self.batch_size        = batch_size
        self.device            = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scale             = scale
        self.metric            = metric
        self.threshold         = threshold

    def run(self) -> float:
        train_records = self.connector.build_train_records()

        groups = np.array([firm_id for firm_id, *_ in train_records])
        gkf = GroupKFold(n_splits=self.k)

        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(np.arange(len(train_records)), groups=groups)
        ):
            train_records_fold = [train_records[i] for i in train_idx]
            val_records_fold   = [train_records[i] for i in val_idx]

            if self.scale:
                scaler             = self.connector.fit_scaler(train_records_fold)
                train_records_fold = self.connector.scale_records(train_records_fold, scaler)
                val_records_fold   = self.connector.scale_records(val_records_fold, scaler)

            train_loader = DataLoader(
                self.dataset_factory(train_records_fold),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
            )
            val_loader = DataLoader(
                self.dataset_factory(val_records_fold),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

            model     = self.model_factory()
            optimizer = self.optimizer_factory(model)
            trainer   = Trainer(model, optimizer, self.criterion, self.device)

            trainer.fit(train_loader, val_loader, n_epochs=self.n_epochs)

            score = trainer.compute_metric(self.metric, val_loader, threshold=self.threshold)
            fold_metrics.append(score)
            print(f"Fold {fold + 1}/{self.k} — val {self.metric}: {score:.4f}")

        mean_score = float(np.mean(fold_metrics))
        print(f"CV mean {self.metric}: {mean_score:.4f}")
        return mean_score
