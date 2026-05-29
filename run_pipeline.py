"""
Main pipeline entrypoint.

Usage:
    uv run python run_pipeline.py --config experiments/baseline.yaml
    uv run python run_pipeline.py --config experiments/baseline.yaml --tune
    uv run python run_pipeline.py --config experiments/baseline.yaml --panel data/simulacion_xyz/panel.csv
"""
import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from rich.logging import RichHandler
from torch.utils.data import DataLoader

from config import DATA_DIR
from connectors.lstm import LSTMConnector
from data_generation.data_config import DATA_CONFIG
from data_generation.simulator import DataSimulator
from models.lstm_classifier import LSTMClassifier
from splits.split_generator import SplitGenerator
from training.trainer import Trainer

log = logging.getLogger(__name__)


def setup_logging(run_dir: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, markup=True),
            logging.FileHandler(run_dir / "run.log"),
        ],
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_data_config(cfg: dict) -> dict:
    data_cfg = copy.deepcopy(DATA_CONFIG)
    data_cfg['n_empresas']        = cfg['data']['n_empresas']
    data_cfg['n_periodos']        = cfg['data']['n_periodos']
    data_cfg['n_cohortes']        = cfg['data']['n_cohortes']
    data_cfg['cupo_por_cohorte']  = cfg['data']['cupo_por_cohorte']
    data_cfg['random_seed']       = cfg['data']['random_seed']
    return data_cfg


def build_dataloaders(
    cfg: dict,
    panel,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, LSTMConnector]:
    split_cfg = cfg['split']
    split_gen = SplitGenerator(
        panel,
        train_nini_ratio=split_cfg['train_nini_ratio'],
        seed=split_cfg['seed'],
    )
    split_gen.generate()

    connector = LSTMConnector(
        panel=panel,
        split=split_gen.split,
        feature_cols=cfg['features'],
    )
    train_ds, test_ds = connector.convert()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
    )
    return train_loader, test_loader, connector


def build_model(cfg: dict, n_features: int, n_cohorts: int) -> LSTMClassifier:
    m = cfg['model']
    return LSTMClassifier(
        n_features=n_features,
        lstm_hidden_size=m['lstm_hidden_size'],
        lstm_num_layers=m['lstm_num_layers'],
        n_cohorts=n_cohorts,
        dropout=m['dropout'],
    )


def train(cfg: dict, panel, run_dir: Path) -> dict:
    t_cfg = cfg['training']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, connector = build_dataloaders(
        cfg, panel, batch_size=t_cfg['batch_size']
    )
    n_cohorts = len(connector._cohorts_periods)
    model = build_model(cfg, n_features=len(connector.feature_cols), n_cohorts=n_cohorts)

    optimizer_cls = getattr(torch.optim, t_cfg['optimizer'])
    optimizer = optimizer_cls(model.parameters(), lr=t_cfg['learning_rate'])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
    )
    trainer.fit(train_loader, test_loader, n_epochs=t_cfg['n_epochs'])

    train_acc = trainer.accuracy(train_loader)
    test_acc  = trainer.accuracy(test_loader)
    log.info(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")

    if cfg['output']['save_model']:
        model_path = run_dir / 'model.pt'
        torch.save(model.state_dict(), model_path)
        log.info(f"Model saved → {model_path}")

    return {'train_accuracy': train_acc, 'test_accuracy': test_acc}


def tune(cfg: dict, panel, run_dir: Path) -> dict:
    import optuna

    opt_cfg   = cfg['optuna']
    t_cfg     = cfg['training']
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ss        = opt_cfg['search_space']

    train_loader, test_loader, connector = build_dataloaders(
        cfg, panel, batch_size=t_cfg['batch_size']
    )
    n_cohorts   = len(connector._cohorts_periods)
    n_features  = len(connector.feature_cols)

    def objective(trial: optuna.Trial) -> float:
        lr               = trial.suggest_categorical('learning_rate', ss['learning_rate'])
        dropout          = trial.suggest_categorical('dropout', ss['dropout'])
        lstm_hidden_size = trial.suggest_categorical('lstm_hidden_size', ss['lstm_hidden_size'])
        lstm_num_layers  = trial.suggest_categorical('lstm_num_layers', ss['lstm_num_layers'])

        model = LSTMClassifier(
            n_features=n_features,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            n_cohorts=n_cohorts,
            dropout=dropout,
        )
        optimizer_cls = getattr(torch.optim, t_cfg['optimizer'])
        optimizer = optimizer_cls(model.parameters(), lr=lr)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.BCEWithLogitsLoss(),
            device=device,
        )
        trainer.fit(train_loader, test_loader, n_epochs=t_cfg['n_epochs'])
        return trainer.accuracy(test_loader)

    study = optuna.create_study(
        direction='maximize',
        storage=f'sqlite:///{run_dir}/optuna.db',
        study_name=cfg['experiment']['name'],
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=opt_cfg['n_trials'])

    best = study.best_trial
    log.info(f"Best trial: accuracy={best.value:.4f} | params={best.params}")

    # Final model with best params
    best_cfg = copy.deepcopy(cfg)
    best_cfg['model']['lstm_hidden_size'] = best.params['lstm_hidden_size']
    best_cfg['model']['lstm_num_layers']  = best.params['lstm_num_layers']
    best_cfg['model']['dropout']          = best.params['dropout']
    best_cfg['training']['learning_rate'] = best.params['learning_rate']

    metrics = train(best_cfg, panel, run_dir)
    metrics['best_optuna_params'] = best.params
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter search')
    parser.add_argument('--panel', default=None, help='Path to existing panel.csv (skips generation)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    run_name = f"{cfg['experiment']['name']}_{int(time.time())}"
    run_dir  = Path(cfg['output']['dir']) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    import shutil
    shutil.copy(args.config, run_dir / 'config.yaml')
    log.info(f"Run [bold]{run_name}[/bold] → {run_dir}")

    # Data
    if args.panel:
        import pandas as pd
        log.info(f"Loading panel from {args.panel}")
        panel = pd.read_csv(args.panel)
    else:
        log.info("Generating panel...")
        data_cfg  = build_data_config(cfg)
        simulator = DataSimulator(data_cfg)
        panel     = simulator.simulate()
        panel.to_csv(run_dir / 'panel.csv', index=False)
        log.info(f"Panel shape: {panel.shape}")

    # Run
    if args.tune:
        metrics = tune(cfg, panel, run_dir)
    else:
        metrics = train(cfg, panel, run_dir)

    metrics_path = run_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Run complete → {run_dir}")


if __name__ == '__main__':
    main()
