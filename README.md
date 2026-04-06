## NPGC

`npgc` is a lightweight Python package for fitting a non-parametric Gaussian copula
to tabular data and generating synthetic samples from the learned distribution.

## Installation

```bash
pip install npgc
```

## Quick Start

```python
import pandas as pd

from npgc import NPGC

df = pd.DataFrame(
    {
        "age": [21, 34, 45, 52],
        "income": [42000, 68000, 91000, 120000],
        "segment": ["A", "B", "B", "C"],
    }
)

model = NPGC()
model.fit(df, random_state=42)

synthetic = model.sample(100, seed=42)
print(synthetic.head())
```

## Features

- Works directly with pandas DataFrames
- Supports numeric and categorical columns
- Preserves cross-column dependence with a Gaussian copula
- Includes model save/load helpers for reuse

## Development

Install development dependencies with:

```bash
uv sync --group dev
```

Run the test suite with:

```bash
.\.venv\Scripts\python -m pytest
```

Build distributions locally with:

```bash
$env:UV_CACHE_DIR='.uv-cache'
uv build
```

## Release

After building, upload the artifacts in `dist/` to PyPI:

```bash
uv publish
```

Or with Twine:

```bash
python -m twine upload dist/*
```
