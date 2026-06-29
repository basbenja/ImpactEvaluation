import logging

import optuna
import torch
import torch.nn as nn

from connectors.base import BaseConnector
from datasets.panel_sequence import PanelSequenceDataset
from models.lstm_classifier import LSTMClassifier
from training.cross_validator import CrossValidator

log = logging.getLogger(__name__)


class Tuner:
    def __init__(
        self,
        connector: BaseConnector,
        n_features: int,
        n_cohorts: int,
        t_cfg: dict,
        opt_cfg: dict,
        device: torch.device,
    ):
        self.connector  = connector
        self.n_features = n_features
        self.n_cohorts  = n_cohorts
        self.t_cfg      = t_cfg
        self.opt_cfg    = opt_cfg
        self.device     = device

    def _objective(self, trial: optuna.Trial) -> float:
        ss = self.opt_cfg['search_space']
        lr               = trial.suggest_categorical('learning_rate', ss['learning_rate'])
        dropout          = trial.suggest_categorical('dropout', ss['dropout'])
        lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', ss['lstm_hidden_size'])
        lstm_num_layers  = trial.suggest_categorical('lstm_num_layers', ss['lstm_num_layers'])

        def model_factory():
            return LSTMClassifier(
                n_features=self.n_features,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                n_cohorts=self.n_cohorts,
                dropout=dropout,
            )

        def optimizer_factory(model):
            cls = getattr(torch.optim, self.t_cfg['optimizer'])
            return cls(model.parameters(), lr=lr)

        cv = CrossValidator(
            connector=self.connector,
            dataset_factory=PanelSequenceDataset,
            model_factory=model_factory,
            optimizer_factory=optimizer_factory,
            criterion=nn.BCEWithLogitsLoss(),
            collate_fn=PanelSequenceDataset.collate_fn,
            k=self.opt_cfg.get('cv_folds', 5),
            n_epochs=self.t_cfg['n_epochs'],
            batch_size=self.t_cfg['batch_size'],
            device=self.device,
            scale=True,
            metric=self.opt_cfg['metric'],
            threshold=self.opt_cfg.get('threshold', 0.5),
        )
        return cv.run()

    def run(self, study_name: str, storage: str) -> optuna.Study:
        study = optuna.create_study(
            direction=self.opt_cfg['direction'],
            storage=storage,
            study_name=study_name,
            load_if_exists=True,
        )
        study.optimize(self._objective, n_trials=self.opt_cfg['n_trials'])

        best   = study.best_trial
        metric = self.opt_cfg['metric']
        log.info(f"Best trial: {metric}={best.value:.4f} | params={best.params}")

        return study
