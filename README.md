# NPGC

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


## Citation

The method underlying this package was first introduced in:

Gabriel Diaz Ramos, Lorenzo Luzi, Debshila Basu Mallick, Richard Baraniuk. *Stable and Privacy-Preserving Synthetic Educational Data with Empirical Marginals: A Copula-Based Approach*. Accepted at EDM 2026 | [Preprint](https://arxiv.org/abs/2604.04195)

BibTeX:

```bibtex
@misc{diazramos2026npgc,
  title={Stable and Privacy-Preserving Synthetic Educational Data with Empirical Marginals: A Copula-Based Approach},
  author={Gabriel Diaz Ramos and Lorenzo Luzi and Debshila Basu Mallick and Richard Baraniuk},
  year={2026},
  note={Accepted at EDM 2026 | Preprint}
}
```

`npgc` is a Python package for fitting a non-parametric Gaussian copula to tabular data and generating synthetic tabular samples from the learned distribution. The implementation combines empirical marginal models with a Gaussian copula dependence structure and includes an optional differential privacy mechanism controlled by `epsilon`.

This package is currently published as version `0.1.0` and should be treated as an alpha-stage research software release.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Minimal Usage](#minimal-usage)
- [Persistence Example](#persistence-example)
- [Technical Notes](#technical-notes)
- [Data Contract](#data-contract)
- [Reproducibility](#reproducibility)
- [Project Metadata](#project-metadata)



## Overview

`NPGC` models each column marginally and then couples the marginals through a Pearson correlation matrix in Gaussian latent space:

1. Each input column is transformed to the unit interval with an empirical CDF.
2. Uniform scores are mapped into Gaussian latent variables with the probit transform.
3. A correlation matrix is estimated in latent space.
4. New latent Gaussian samples are drawn and inverse-transformed back into the original feature domains.

The implementation supports:

- numeric columns
- integer-valued numeric columns
- categorical columns
- mixed-type `pandas.DataFrame` inputs
- missing values through column-wise missingness estimation
- model persistence with `save(...)` and `load(...)`
- optional differential privacy noise through `epsilon`

## Requirements

- Python `>=3.10`
- NumPy
- pandas
- SciPy

## Installation

### PyPI

```bash
python -m pip install -U npgc
```

### Google Colab

```python
!python -m pip install -U --no-cache-dir npgc
from npgc import NPGC
```

### VS Code with `uv`

Add the package to the active project environment:

```bash
uv add npgc
```

Then import it normally:

```python
from npgc import NPGC
```

Constructor signature:

```python
NPGC(enforce_min_max_values: bool = True, epsilon: float | None = 1.0)
```

### `NPGC.__init__(enforce_min_max_values=True, epsilon=1.0)`

Initializes an unfitted synthesizer.

| Parameter | Type | Default | Technical meaning |
| --- | --- | --- | --- |
| `enforce_min_max_values` | `bool` | `True` | Controls tail behavior during inverse ECDF reconstruction. When `True`, continuous outputs remain within the observed training range and integer outputs are snapped to the observed integer support. When `False`, continuous and integer-valued variables may extrapolate beyond the observed extrema. |
| `epsilon` | `float \| None` | `1.0` | Default differential privacy budget used during `fit(...)` if no per-fit override is provided. If `None` or non-positive, the privacy mechanism is disabled and empirical statistics are used directly. |

### `fit(data, epsilon=None, random_state=None)`

Method signature:

```python
fit(data: pandas.DataFrame, epsilon: float | None = None, random_state: int | None = None) -> None
```

Fits the synthesizer to a tabular dataset.

| Parameter | Type | Default | Technical meaning |
| --- | --- | --- | --- |
| `data` | `pandas.DataFrame` | required | Training table. The implementation requires a non-empty `DataFrame`. |
| `epsilon` | `float \| None` | `None` | Optional fit-time override for the instance privacy budget. If supplied, it takes precedence over `self.epsilon`. |
| `random_state` | `int \| None` | `None` | Seed used for reproducible privacy noise and randomized empirical CDF tie-breaking during fitting. |

Behavior:

- Raises `ValueError` if `data` is not a `pandas.DataFrame`.
- Raises `ValueError` if `data` is empty.
- Stores learned marginals, latent correlation matrix, and column order internally.
- Marks the model as fitted.

### `sample(num_rows, seed=None)`

Method signature:

```python
sample(num_rows: int, seed: int | None = None) -> pandas.DataFrame
```

Generates synthetic rows from a previously fitted model.

| Parameter | Type | Default | Technical meaning |
| --- | --- | --- | --- |
| `num_rows` | `int` | required | Number of synthetic rows to generate. |
| `seed` | `int \| None` | `None` | Random seed for reproducible sampling from the latent Gaussian model. |

Behavior:

- Raises `RuntimeError` if called before `fit(...)`.
- Returns a `pandas.DataFrame` with the learned column order.
- Attempts to cast each generated column back to the original training dtype.

### `save(filepath)`

Method signature:

```python
save(filepath: str | os.PathLike[str]) -> None
```

Serializes the fitted model as a pickle file. Parent directories are created automatically when needed.

### `load(filepath)`

Method signature:

```python
load(filepath: str | os.PathLike[str]) -> None
```

Loads model state into the current `NPGC` instance from a pickle file. The loader supports both object-based checkpoints and a legacy dictionary-based state format.

## Minimal Usage

```python
import pandas as pd

from npgc import NPGC

df = pd.DataFrame(
    {
        "age": [21, 34, 45, 52],
        "income": [42000.0, 68000.0, 91000.0, 120000.0],
        "segment": ["A", "B", "B", "C"],
    }
)

model = NPGC(enforce_min_max_values=True, epsilon=1.0)
model.fit(df, random_state=42)

synthetic = model.sample(100, seed=42)
print(synthetic.head())
```

## Persistence Example

```python
from npgc import NPGC

model = NPGC(epsilon=1.0)
model.fit(df, random_state=42)
model.save("artifacts/npgc_model.pkl")

reloaded = NPGC()
reloaded.load("artifacts/npgc_model.pkl")
synthetic = reloaded.sample(50, seed=7)
```

## Technical Notes

### What `epsilon` does

`epsilon` is the differential privacy budget.

- Smaller `epsilon` means stronger privacy and more perturbation.
- Larger `epsilon` means weaker privacy and less perturbation.
- `epsilon=None` disables the privacy mechanism.
- `epsilon<=0` is treated as non-private in the current implementation.

The current implementation splits the total privacy budget equally:

- `epsilon / 2` for marginal estimation
- `epsilon / 2` for latent correlation estimation

Mechanistically, the code applies Laplace noise to:

- integer support counts
- continuous histograms
- categorical counts
- latent-space correlation estimates

This means privacy is not added only at the final sample stage; it is injected during model fitting into the sufficient statistics used to construct the synthetic generator.

### What `enforce_min_max_values` does

`enforce_min_max_values` controls whether inverse marginal reconstruction is range-constrained.

When `True`:

- continuous columns are reconstructed within the observed training range
- integer columns are mapped to the nearest observed integer support value
- generated values remain conservative with respect to the observed empirical support

When `False`:

- continuous columns may extrapolate beyond the observed minimum and maximum
- integer columns may extrapolate before final dtype casting
- the model can generate values outside the original empirical range

This parameter is especially important when synthetic outputs must remain support-faithful for downstream validation, auditing, or schema-constrained pipelines.

### Missing values

Missingness is modeled per column through the observed missing fraction:

- numeric columns preserve an estimated `nan_frac`
- categorical columns reserve probability mass for missing values
- synthetic samples may therefore contain missing values when the training data does

### Type handling

Column handling is determined from the training `DataFrame`:

- numeric dtypes are modeled as either integer or continuous
- non-numeric dtypes are treated as categorical
- output columns are cast back toward the original dtype after generation

For integer-valued numeric columns, the implementation detects integer structure from the observed non-missing values and uses a dedicated inverse ECDF path.

## Data Contract

Expected input:

- a non-empty `pandas.DataFrame`
- tabular columns with numeric or categorical-like values
- optional missing values

Current implementation details worth knowing:

- column order is preserved
- correlations are computed in Gaussian latent space with Pearson correlation
- the correlation matrix is repaired to the nearest valid correlation matrix when noise or numerical issues make it non-PSD
- categorical labels are sampled from observed label support

## Reproducibility

There are two independent random entry points:

- `random_state` in `fit(...)` controls fitting-time randomness, including privacy noise and randomized ECDF operations
- `seed` in `sample(...)` controls synthetic sample generation after fitting

For exact reproducibility, set both.

## Project Metadata

- Package name: `npgc`
- Current version: `0.1.0`
- Issue tracker: <https://github.com/gdiaz95/NPGC/issues>
