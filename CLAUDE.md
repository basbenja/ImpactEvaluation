# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Managed with `uv` (Python 3.12).

```bash
uv sync          # creates venv + installs deps
```

Environment variable `DATA_DIR` controls where panel CSVs land (defaults to `./data/`). Copy `.env.template` to `.env` and set if needed.

## Running

Main workflow lives in `main.ipynb`. Data generation is in `data_generation/main.py` (or its notebook counterpart). Run notebooks with the `ipykernel` dev dependency already installed.

To generate synthetic panel data:

```bash
uv run python data_generation/main.py
```

Outputs land in `data/simulacion_<timestamp>/panel.csv` + `config.json`.

For hyperparameter optimization (Optuna), the dashboard is available via `optuna-dashboard`.

## Architecture

This project uses an LSTM-based classifier to identify counterfactual control firms for **staggered treatment impact evaluation**. Three firm types: **Tratados (T)**, **Controles (C)**, **NiNis** (neither treated nor control).

### Data flow

```
data_generation/
  data_config.py     ← single DATA_CONFIG dict with all DGP parameters
  simulator.py       ← DataSimulator: generates long-format panel DataFrame

splits/
  split_generator.py ← SplitGenerator: partitions firms into train/test
                        train = all T + fraction of NiNi
                        test  = all C + remaining NiNi

connectors/
  lstm.py            ← LSTMConnector: panel + split → PanelSequenceDataset
                        builds variable-length pre-treatment sequences per firm

models/
  blocks/lstm.py     ← LSTMBlock: pack_padded_sequence to handle variable lengths
  blocks/dense.py    ← DenseBlock: linear layers + ReLU + Dropout → scalar logit
  lstm_classifier.py ← LSTMClassifier: LSTM hidden state + one-hot cohort → binary label

training/
  trainer.py         ← Trainer: fit/validation loop + accuracy
```

### Key design decisions

**Train set**: all T firms (label=1) + subset of NiNi firms repeated once per cohort (label=0). NiNis are repeated so the model learns they are non-participants for *every* cohort.

**Test set**: all C firms repeated once per cohort — label=1 for their actual cohort, label=0 for others — plus remaining NiNi (label=0). The goal is to identify controls in their true cohort.

**Variable-length sequences**: each record is pre-treatment observations up to `t_k` (cohort start). `LSTMBlock` uses `pack_padded_sequence`/`pad_packed_sequence` to skip padded timesteps. `collate_fn` in `PanelSequenceDataset` handles batching.

**Cohort as feature**: cohort id is one-hot encoded and concatenated with the LSTM's final hidden state before the dense head. This lets the model condition its prediction on which cohort window is being evaluated.

**Scaler**: `LSTMConnector.convert()` fits a `StandardScaler` on flattened train sequences and applies it to both splits. Scaler is stored in `self.scaler` for reuse.

### Panel format

Long format: one row per `(id_firma, t)`. Key columns:

| Column | Meaning |
|---|---|
| `id_firma` | firm ID |
| `t` | period index |
| `tratado` | True if firm received treatment (ever) and `t >= periodo_tratamiento` |
| `control` | True if firm is a counterfactual control |
| `cohorte` | cohort index (0-based); -1 if not treated |
| `periodo_tratamiento` | period when treatment started; -1 if never |
| `inicio_firma` | first period the firm exists |

Firms with `tratado=False, control=False` are NiNis.

### Data generation parameters

All DGP parameters live in `data_generation/data_config.py` as `DATA_CONFIG`. Key sections:
- `variables`: distributions for observable + unobservable firm characteristics
- `dinamica_outcomes`: AR(1) dynamics for `empleados` and `salario_promedio`
- `elegibilidad`: deterministic eligibility rules
- `seleccion`: logit model for treatment propensity (includes unobservable confounders)
- `efectos_tratamiento`: dynamic, heterogeneous treatment effects
- `efecto_demostracion`: spillover from prior cohorts' visible outcomes
