import pandas as pd

from npgc import NPGC, __version__


def test_fit_and_sample_round_trip() -> None:
    df = pd.DataFrame(
        {
            "age": [21, 34, 45, 52],
            "income": [42000.0, 68000.0, 91000.0, 120000.0],
            "segment": ["A", "B", "B", "C"],
        }
    )

    model = NPGC()
    model.fit(df, random_state=42)
    sampled = model.sample(8, seed=42)

    assert list(sampled.columns) == list(df.columns)
    assert len(sampled) == 8


def test_version_is_defined() -> None:
    assert isinstance(__version__, str)
    assert __version__
