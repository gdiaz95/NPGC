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

Build distributions locally with:

```bash
python -m build
```
