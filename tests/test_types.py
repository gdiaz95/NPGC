"""Comprehensive pytest tests for all NPGC variable types:
continuous, integer, categorical, datetime.

Run: pytest tests/test_types.py -v
"""

import numpy as np
import pandas as pd
import pytest
from npgc import NPGC

RNG = np.random.default_rng(42)
N = 600  # enough rows for stable DP results without being slow


# ─── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def continuous_df():
    vals = RNG.exponential(5, N)
    return pd.DataFrame({"x": vals, "y": RNG.normal(0, 1, N)})


@pytest.fixture
def integer_df():
    ages = RNG.integers(18, 80, N)
    counts = RNG.integers(0, 50, N)
    return pd.DataFrame({"age": ages, "count": counts})


@pytest.fixture
def categorical_df():
    labels = RNG.choice(["A", "B", "C", "D"], N, p=[0.4, 0.3, 0.2, 0.1])
    tier = RNG.choice(["low", "mid", "high"], N, p=[0.5, 0.3, 0.2])
    return pd.DataFrame({"label": labels, "tier": tier})


@pytest.fixture
def datetime_df():
    dates = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        RNG.exponential(400, N).astype(int), unit="D"
    )
    return pd.DataFrame({"date": dates, "value": RNG.normal(0, 1, N)})


@pytest.fixture
def mixed_df():
    dates = pd.to_datetime("2022-03-01") + pd.to_timedelta(
        RNG.integers(0, 730, N), unit="D"
    )
    return pd.DataFrame({
        "date": dates,
        "age": RNG.integers(18, 70, N),
        "score": RNG.normal(100, 15, N),
        "plan": RNG.choice(["free", "pro", "enterprise"], N, p=[0.6, 0.3, 0.1]),
    })


def _fit_sample(df, epsilon=None, n=None, enforce=True, seed=0):
    m = NPGC(epsilon=epsilon, enforce_min_max_values=enforce)
    m.fit(df, random_state=0)
    return m, m.sample(n or len(df), seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# CONTINUOUS
# ═══════════════════════════════════════════════════════════════════════════

class TestContinuous:

    def test_meta_type(self, continuous_df):
        m, _ = _fit_sample(continuous_df)
        for col in continuous_df.columns:
            assert m._model_state["marginals"][col]["type"] == "continuous"

    def test_dtype_preserved(self, continuous_df):
        _, synth = _fit_sample(continuous_df)
        for col in continuous_df.columns:
            assert synth[col].dtype == continuous_df[col].dtype

    def test_column_order(self, continuous_df):
        _, synth = _fit_sample(continuous_df)
        assert list(synth.columns) == list(continuous_df.columns)

    def test_row_count(self, continuous_df):
        _, synth = _fit_sample(continuous_df, n=250)
        assert len(synth) == 250

    def test_no_nan_when_none_in_input(self, continuous_df):
        _, synth = _fit_sample(continuous_df)
        assert synth.notna().all().all()

    def test_enforce_min_max(self, continuous_df):
        _, synth = _fit_sample(continuous_df, enforce=True)
        for col in continuous_df.columns:
            assert synth[col].min() >= continuous_df[col].min()
            assert synth[col].max() <= continuous_df[col].max()

    def test_nan_fraction_preserved(self):
        vals = RNG.normal(0, 1, N).astype(float)
        vals[RNG.random(N) < 0.2] = np.nan
        df = pd.DataFrame({"x": vals})
        orig_nf = df["x"].isna().mean()
        _, synth = _fit_sample(df, n=2000)
        synth_nf = synth["x"].isna().mean()
        assert abs(synth_nf - orig_nf) < 0.05, f"orig={orig_nf:.3f} synth={synth_nf:.3f}"

    def test_all_nan_column(self):
        df = pd.DataFrame({"x": np.full(N, np.nan), "y": RNG.normal(0, 1, N)})
        m, synth = _fit_sample(df)
        assert synth["x"].isna().all()

    def test_constant_column(self):
        df = pd.DataFrame({"x": np.full(N, 3.14), "y": RNG.normal(0, 1, N)})
        _, synth = _fit_sample(df)
        assert synth["x"].notna().any()

    def test_no_dp_anchors_equal_originals(self, continuous_df):
        m, _ = _fit_sample(continuous_df, epsilon=None)
        meta = m._model_state["marginals"]["x"]
        orig = np.sort(continuous_df["x"].dropna().values)
        assert np.allclose(meta["sorted_values"], orig)

    def test_dp_anchors_differ_from_originals(self, continuous_df):
        m, _ = _fit_sample(continuous_df, epsilon=1.0)
        meta = m._model_state["marginals"]["x"]
        orig_set = set(continuous_df["x"].dropna().values)
        anchor_set = set(meta["sorted_values"])
        assert len(anchor_set & orig_set) == 0, "DP anchors should not match original float values"

    def test_values_are_finite(self, continuous_df):
        _, synth = _fit_sample(continuous_df)
        for col in continuous_df.columns:
            assert np.isfinite(synth[col].dropna()).all()


# ═══════════════════════════════════════════════════════════════════════════
# INTEGER
# ═══════════════════════════════════════════════════════════════════════════

class TestInteger:

    def test_meta_type(self, integer_df):
        m, _ = _fit_sample(integer_df)
        for col in integer_df.columns:
            assert m._model_state["marginals"][col]["type"] == "integer"

    def test_dtype_preserved(self, integer_df):
        _, synth = _fit_sample(integer_df)
        for col in integer_df.columns:
            assert "int" in str(synth[col].dtype).lower()

    def test_output_values_are_integers(self, integer_df):
        _, synth = _fit_sample(integer_df)
        for col in integer_df.columns:
            valid = synth[col].dropna()
            assert np.allclose(valid % 1, 0), f"{col} has non-integer values"

    def test_enforce_within_observed_support(self, integer_df):
        m, synth = _fit_sample(integer_df, enforce=True)
        for col in integer_df.columns:
            uniques = integer_df[col].unique()
            assert synth[col].dropna().isin(uniques).all(), \
                f"{col} has values outside observed support"

    def test_no_nan_when_none_in_input(self, integer_df):
        _, synth = _fit_sample(integer_df)
        assert synth.notna().all().all()

    def test_nan_fraction_preserved(self):
        vals = RNG.integers(1, 10, N).astype(float)
        vals[RNG.random(N) < 0.15] = np.nan
        df = pd.DataFrame({"x": pd.array(vals, dtype="Float64")})
        df["x"] = df["x"].where(~np.isnan(vals))
        orig_nf = df["x"].isna().mean()
        _, synth = _fit_sample(df, n=2000)
        synth_nf = synth["x"].isna().mean()
        assert abs(synth_nf - orig_nf) < 0.05, f"orig={orig_nf:.3f} synth={synth_nf:.3f}"

    def test_column_order(self, integer_df):
        _, synth = _fit_sample(integer_df)
        assert list(synth.columns) == list(integer_df.columns)

    def test_row_count(self, integer_df):
        _, synth = _fit_sample(integer_df, n=333)
        assert len(synth) == 333

    def test_single_unique_value(self):
        df = pd.DataFrame({"x": np.full(N, 7, dtype=np.int64)})
        _, synth = _fit_sample(df)
        valid = synth["x"].dropna()
        assert (valid == 7).all(), "Single-value integer should reproduce that value"

    def test_dp_does_not_crash(self, integer_df):
        _, synth = _fit_sample(integer_df, epsilon=0.5)
        assert len(synth) == len(integer_df)


# ═══════════════════════════════════════════════════════════════════════════
# CATEGORICAL
# ═══════════════════════════════════════════════════════════════════════════

class TestCategorical:

    def test_meta_type(self, categorical_df):
        m, _ = _fit_sample(categorical_df)
        for col in categorical_df.columns:
            assert m._model_state["marginals"][col]["type"] == "categorical"

    def test_only_observed_labels(self, categorical_df):
        _, synth = _fit_sample(categorical_df)
        for col in categorical_df.columns:
            observed = set(categorical_df[col].dropna().unique())
            synthetic_vals = set(synth[col].dropna().unique())
            assert synthetic_vals.issubset(observed), \
                f"{col}: unexpected labels {synthetic_vals - observed}"

    def test_all_observed_labels_appear(self, categorical_df):
        _, synth = _fit_sample(categorical_df, n=2000)
        for col in categorical_df.columns:
            observed = set(categorical_df[col].dropna().unique())
            synthetic_vals = set(synth[col].dropna().unique())
            assert observed == synthetic_vals, \
                f"{col}: missing labels {observed - synthetic_vals}"

    def test_dtype_preserved(self, categorical_df):
        _, synth = _fit_sample(categorical_df)
        for col in categorical_df.columns:
            assert synth[col].dtype == categorical_df[col].dtype

    def test_no_nan_when_none_in_input(self, categorical_df):
        _, synth = _fit_sample(categorical_df)
        assert synth.notna().all().all()

    def test_nan_fraction_preserved(self):
        vals = RNG.choice(["X", "Y", "Z"], N).astype(object)
        vals[RNG.random(N) < 0.18] = None
        df = pd.DataFrame({"cat": pd.Series(vals, dtype=object)})
        orig_nf = df["cat"].isna().mean()
        _, synth = _fit_sample(df, n=2000)
        synth_nf = synth["cat"].isna().mean()
        assert abs(synth_nf - orig_nf) < 0.05, f"orig={orig_nf:.3f} synth={synth_nf:.3f}"

    def test_all_nan_column(self):
        df = pd.DataFrame({
            "cat": pd.Series([None] * N, dtype=object),
            "num": RNG.normal(0, 1, N),
        })
        _, synth = _fit_sample(df)
        assert synth["cat"].isna().all()

    def test_single_label(self):
        df = pd.DataFrame({"cat": pd.Series(["only"] * N, dtype=object)})
        _, synth = _fit_sample(df)
        assert (synth["cat"].dropna() == "only").all()

    def test_frequency_approximately_preserved(self):
        probs = np.array([0.5, 0.3, 0.2])
        labels = RNG.choice(["A", "B", "C"], N * 5, p=probs)
        df = pd.DataFrame({"cat": pd.Series(labels, dtype=object)})
        _, synth = _fit_sample(df, epsilon=None, n=N * 5)
        for label, expected_p in zip(["A", "B", "C"], probs):
            actual_p = (synth["cat"] == label).mean()
            assert abs(actual_p - expected_p) < 0.05, \
                f"label {label}: expected {expected_p:.2f}, got {actual_p:.2f}"

    def test_dp_labels_unchanged(self, categorical_df):
        m, _ = _fit_sample(categorical_df, epsilon=1.0)
        for col in categorical_df.columns:
            meta = m._model_state["marginals"][col]
            orig_labels = sorted(categorical_df[col].dropna().unique().tolist())
            assert meta["labels"] == orig_labels

    def test_column_order(self, categorical_df):
        _, synth = _fit_sample(categorical_df)
        assert list(synth.columns) == list(categorical_df.columns)


# ═══════════════════════════════════════════════════════════════════════════
# DATETIME
# ═══════════════════════════════════════════════════════════════════════════

class TestDatetime:

    def test_meta_type(self, datetime_df):
        m, _ = _fit_sample(datetime_df)
        assert m._model_state["marginals"]["date"]["type"] == "datetime"

    def test_dtype_preserved_tz_naive(self, datetime_df):
        _, synth = _fit_sample(datetime_df)
        assert pd.api.types.is_datetime64_any_dtype(synth["date"])
        assert synth["date"].dt.tz is None

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Europe/Berlin", "Asia/Tokyo"])
    def test_dtype_preserved_tz_aware(self, tz):
        ts = pd.date_range("2023-01-01", periods=N, freq="3h", tz=tz)
        df = pd.DataFrame({"ts": ts})
        _, synth = _fit_sample(df)
        assert str(synth["ts"].dtype) == str(df["ts"].dtype), \
            f"Expected {df['ts'].dtype}, got {synth['ts'].dtype}"

    @pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Europe/Berlin", "Asia/Tokyo"])
    def test_tz_stored_in_meta(self, tz):
        ts = pd.date_range("2023-01-01", periods=N, freq="3h", tz=tz)
        df = pd.DataFrame({"ts": ts})
        m, _ = _fit_sample(df)
        assert m._model_state["marginals"]["ts"]["tz"] == str(tz)

    def test_enforce_min_max(self, datetime_df):
        _, synth = _fit_sample(datetime_df, enforce=True)
        assert synth["date"].dropna().min() >= datetime_df["date"].min()
        assert synth["date"].dropna().max() <= datetime_df["date"].max()

    def test_no_nat_when_none_in_input(self, datetime_df):
        _, synth = _fit_sample(datetime_df)
        assert synth["date"].notna().all()

    def test_nan_fraction_all_nat(self):
        df = pd.DataFrame({
            "date": pd.DatetimeIndex([pd.NaT] * N),
            "x": RNG.normal(0, 1, N),
        })
        m, synth = _fit_sample(df)
        meta = m._model_state["marginals"]["date"]
        assert abs(meta["nan_frac"] - 1.0) < 1e-9
        assert len(meta["sorted_values"]) == 0
        assert synth["date"].isna().all()

    def test_nan_fraction_partial_nat(self):
        target = 0.20
        dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(
            RNG.integers(0, 365, N), unit="D"
        )
        arr = dates.to_numpy(dtype="datetime64[ns]").copy()
        arr[RNG.random(N) < target] = np.datetime64("NaT")
        df = pd.DataFrame({"date": pd.DatetimeIndex(arr), "x": RNG.normal(0, 1, N)})
        orig_nf = df["date"].isna().mean()
        _, synth = _fit_sample(df, n=2000)
        synth_nf = synth["date"].isna().mean()
        assert abs(synth_nf - orig_nf) < 0.05, f"orig={orig_nf:.3f} synth={synth_nf:.3f}"

    def test_no_dp_anchors_exact_originals(self, datetime_df):
        m, _ = _fit_sample(datetime_df, epsilon=None)
        meta = m._model_state["marginals"]["date"]
        orig_secs = set(
            (datetime_df["date"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
        )
        anchors = set(meta["sorted_values"])
        assert anchors == orig_secs

    def test_dp_anchors_differ_from_originals(self, datetime_df):
        m, _ = _fit_sample(datetime_df, epsilon=1.0)
        meta = m._model_state["marginals"]["date"]
        orig_secs = set(
            (datetime_df["date"] - pd.Timestamp("1970-01-01")).dt.total_seconds()
        )
        anchors = set(meta["sorted_values"])
        assert len(anchors & orig_secs) == 0, \
            "DP anchors must not reproduce original timestamps"

    def test_constant_datetime(self):
        const = pd.Timestamp("2023-07-15 12:00:00")
        df = pd.DataFrame({"date": pd.DatetimeIndex([const] * N)})
        m, synth = _fit_sample(df)
        meta = m._model_state["marginals"]["date"]
        assert abs(meta["nan_frac"]) < 1e-9
        assert synth["date"].notna().any()

    def test_column_order(self, datetime_df):
        _, synth = _fit_sample(datetime_df)
        assert list(synth.columns) == list(datetime_df.columns)

    def test_row_count(self, datetime_df):
        _, synth = _fit_sample(datetime_df, n=123)
        assert len(synth) == 123

    def test_output_is_datetime_type(self, datetime_df):
        _, synth = _fit_sample(datetime_df)
        assert pd.api.types.is_datetime64_any_dtype(synth["date"])

    def test_intraday_resolution(self):
        base = pd.Timestamp("2024-06-01")
        ts = pd.DatetimeIndex([
            base + pd.Timedelta(hours=float(h), minutes=int(m))
            for h, m in zip(RNG.uniform(8, 17, N), RNG.integers(0, 60, N))
        ])
        df = pd.DataFrame({"ts": ts})
        _, synth = _fit_sample(df)
        # synthetic timestamps should also fall within the same day
        assert (synth["ts"].dropna().dt.date == base.date()).all()


# ═══════════════════════════════════════════════════════════════════════════
# MIXED — all types together
# ═══════════════════════════════════════════════════════════════════════════

class TestMixed:

    def test_all_types_fit_and_sample(self, mixed_df):
        _, synth = _fit_sample(mixed_df)
        assert list(synth.columns) == list(mixed_df.columns)
        assert len(synth) == len(mixed_df)

    def test_dtypes_all_preserved(self, mixed_df):
        _, synth = _fit_sample(mixed_df)
        assert pd.api.types.is_datetime64_any_dtype(synth["date"])
        assert "int" in str(synth["age"].dtype).lower()
        assert synth["plan"].dtype == mixed_df["plan"].dtype

    def test_no_unexpected_nans(self, mixed_df):
        _, synth = _fit_sample(mixed_df)
        assert synth.notna().all().all()

    def test_categorical_labels_valid(self, mixed_df):
        _, synth = _fit_sample(mixed_df)
        observed = set(mixed_df["plan"].unique())
        assert set(synth["plan"].dropna().unique()).issubset(observed)

    def test_integer_values_are_integers(self, mixed_df):
        _, synth = _fit_sample(mixed_df)
        assert np.allclose(synth["age"].dropna() % 1, 0)

    def test_datetime_within_range(self, mixed_df):
        _, synth = _fit_sample(mixed_df, enforce=True)
        assert synth["date"].dropna().min() >= mixed_df["date"].min()
        assert synth["date"].dropna().max() <= mixed_df["date"].max()

    def test_correlation_preserved(self, mixed_df):
        """date↔score Pearson correlation should be approximately preserved."""
        df = mixed_df.copy()
        # build a meaningful correlation
        days = (df["date"] - df["date"].min()).dt.days.values
        df["score"] = 50 + 0.3 * days + RNG.normal(0, 30, N)

        _, synth = _fit_sample(df, epsilon=None, n=N)
        orig_r = df["date"].astype(np.int64).corr(df["score"])
        synth_r = synth["date"].astype(np.int64).corr(synth["score"])
        assert abs(orig_r - synth_r) < 0.15, f"orig={orig_r:.3f} synth={synth_r:.3f}"

    def test_dp_full_pipeline(self, mixed_df):
        """Full DP pipeline must not crash and must produce correct shapes."""
        _, synth = _fit_sample(mixed_df, epsilon=1.0)
        assert synth.shape == mixed_df.shape

    def test_save_load_roundtrip(self, mixed_df, tmp_path):
        m, synth_before = _fit_sample(mixed_df, seed=7)
        path = tmp_path / "model.pkl"
        m.save(path)

        m2 = NPGC()
        m2.load(path)
        synth_after = m2.sample(len(mixed_df), seed=7)

        pd.testing.assert_frame_equal(synth_before, synth_after)
