# Impact Evaluation

Authors:
- Bas Peralta, Benjamín
- Giuliodori, David Augusto
- Domínguez, Martín Ariel
- Rodríguez, Alejandro

# Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for managing
    virtual environments and Python versions.

# Install dependencies
**Having installed `uv`, `cd` into the project directory and run:**
```bash
uv sync
```
This command will automatically create a virtual environment and install al the
dependencies declared in `pyproject.toml`.

# Modules
This repository consists of the following modules:
- `data_generation`: Responsible for generating synthetic data for impact evaluation.

## `data_generation`
The configuration for data generation is specified in the
`data_generation/config.py` file.

To generate synthetic data, run the `data_generation/main.ipynb` notebook.