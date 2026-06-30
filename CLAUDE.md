# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Managed with `uv` (Python 3.12).

```bash
uv sync          # creates venv + installs deps
```

Environment variable `DATA_DIR` controls where panel CSVs land (defaults to `./data/`). Copy `.env.template` to `.env` and set if needed.

## Running

**Preferred entrypoint** is `run_pipeline.py`:

```bash
uv run python run_pipeline.py --config experiments/exp_000.yaml          # train
uv run python run_pipeline.py --config experiments/exp_000.yaml --tune   # Optuna search
uv run python run_pipeline.py --config experiments/exp_000.yaml --panel data/simulacion_xyz/panel.csv  # skip data gen
```

Each run creates `runs/<name>_<timestamp>/` containing `config.yaml`, `panel.csv`, `model.pt`, `metrics.json`, and `run.log`. Optuna runs also write `optuna.db`.

```bash
optuna-dashboard sqlite:///runs/<run_dir>/optuna.db   # view Optuna results
```

Experiment configs live in `experiments/`. Named after Greek letters or `exp_NNN`.

To generate data standalone:

```bash
uv run python data_generation/main.py
```

Outputs land in `data/simulacion_<timestamp>/panel.csv` + `config.json`.

## Architecture

LSTM-based classifier to identify counterfactual control firms for **staggered treatment impact evaluation**. Three firm types: **Tratados (T)**, **Controles (C)**, **NiNis** (neither treated nor control).

### Data flow

```
data_generation/
  data_config.py     ← single DATA_CONFIG dict with all DGP parameters
  simulator.py       ← DataSimulator: generates long-format panel DataFrame
  panel_schema.py    ← Col (column name constants) + PanelSchema (pandera validation)

splits/
  split_generator.py ← SplitGenerator: partitions firms into train/test
                        train = all T + fraction of NiNi
                        test  = all C + remaining NiNi

connectors/
  base.py            ← BaseConnector (ABC) + Record type alias + preprocess_panel()
  lstm.py            ← LSTMConnector: panel + split → PanelSequenceDataset
                        builds variable-length pre-treatment sequences per firm

datasets/
  panel_sequence.py  ← PanelSequenceDataset: wraps list[Record]; collate_fn pads
                        sequences and returns (firm_ids, seqs_padded, lengths, cohorts, labels)

models/
  blocks/lstm.py     ← LSTMBlock: pack_padded_sequence to handle variable lengths
  blocks/dense.py    ← DenseBlock: linear layers + ReLU + Dropout → scalar logit
  lstm_classifier.py ← LSTMClassifier: LSTM hidden state + one-hot cohort → binary label

training/
  trainer.py         ← Trainer: fit/eval loop; metrics: accuracy, f1, precision, recall, roc_auc
  cross_validator.py ← CrossValidator: GroupKFold on train records (groups by firm_id)
  tuner.py           ← Tuner: Optuna study wrapping CrossValidator; categorical search space

experiments/         ← YAML configs; one file per experiment run
run_pipeline.py      ← CLI entrypoint: data gen → split → connector → train/tune → save
```

### Key design decisions

**Train set**: all T firms (label=1) + subset of NiNi firms repeated once per cohort (label=0). NiNis are repeated so the model learns they are non-participants for *every* cohort.

**Test set**: all C firms repeated once per cohort — label=1 for their actual cohort, label=0 for others — plus remaining NiNi (label=0). The goal is to identify controls in their true cohort.

**Variable-length sequences**: each record is pre-treatment observations up to `t_k` (cohort start). `LSTMBlock` uses `pack_padded_sequence`/`pad_packed_sequence` to skip padded timesteps. `collate_fn` in `PanelSequenceDataset` handles batching.

**Cohort as feature**: cohort id is one-hot encoded and concatenated with the LSTM's final hidden state before the dense head. This lets the model condition its prediction on which cohort window is being evaluated.

**Feature encoding**: `BaseConnector.preprocess_panel()` splits features by dtype — numeric columns are left for `StandardScaler`; bool/binary-int columns cast to float32 and passed through unscaled; object/categorical columns become one-hot dummies (unscaled). Column order after expansion: `[numeric | indicators | dummies]`. `n_numeric` tracks the boundary for partial scaling.

**Scaler**: `LSTMConnector.convert()` fits a `StandardScaler` on flattened train sequences (numeric cols only) and applies it to both splits. Scaler stored in `self.scaler` for reuse. `CrossValidator` re-fits a fresh scaler per fold when `scale=True`.

**Cross-validation**: `CrossValidator` uses `GroupKFold` grouped by `firm_id` so the same firm never appears in both train and val folds.

**Tuner**: `Tuner.run()` creates (or resumes) an Optuna study backed by SQLite. Each trial builds a `CrossValidator` with the sampled hyperparams and returns the mean CV metric. Search space is categorical (defined in `optuna.search_space` in the YAML). Available metrics for `optuna.metric`: `accuracy`, `f1`, `precision`, `recall`, `roc_auc`.

**Logging**: `run_pipeline.py` uses `logging` + `RichHandler` (colored console) + `FileHandler` (`run.log`). Use `log = logging.getLogger(__name__)` in any new module; the root logger is configured in `setup_logging()` at run start.

### Panel format

Long format: one row per `(id_firma, t)`. Use `Col` constants from `data_generation/panel_schema.py` for column names. Key columns:

| Column | Meaning |
|---|---|
| `id_firma` | firm ID |
| `t` | period index |
| `tratado_en_t` | True if firm is treated **at this period** (`t >= periodo_tratamiento`) |
| `control_en_t` | True if firm is a counterfactual control **at this period** |
| `cohorte` | cohort index (0-based); -1 if not treated |
| `periodo_tratamiento` | period when treatment started; -1 if never |
| `inicio_firma` | first period the firm exists |

Firms with `tratado_en_t=False` in all periods and `control_en_t=False` in all periods are NiNis.

### Data generation parameters

All DGP parameters live in `data_generation/data_config.py` as `DATA_CONFIG`. Key sections:
- `variables`: distributions for observable + unobservable firm characteristics
- `dinamica_outcomes`: AR(1) dynamics for `empleados` and `salario_promedio`
- `elegibilidad`: deterministic eligibility rules
- `seleccion`: logit model for treatment propensity (includes unobservable confounders)
- `efectos_tratamiento`: dynamic, heterogeneous treatment effects
- `efecto_demostracion`: spillover from prior cohorts' visible outcomes
