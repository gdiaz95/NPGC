"""Intensive datetime tests for NPGC v0.2.0.

Run: python test_datetime_npgc.py
Produces a multi-page PDF: datetime_test_report.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from npgc import NPGC

PDF_PATH = "datetime_test_report.pdf"
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

rng = np.random.default_rng(0)
failures = []


# ─── helpers ────────────────────────────────────────────────────────────────

def check(name: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(name)


def plot_datetime_marginal(ax, orig, synth, title: str, date_fmt: str = "%Y-%m") -> None:
    """Overlay histogram of original vs synthetic datetime column."""
    valid_orig = orig.dropna()
    valid_synth = synth.dropna()
    if valid_orig.empty:
        ax.set_title(f"{title}\n(all NaT)")
        ax.text(0.5, 0.5, "all NaT", ha="center", va="center", transform=ax.transAxes)
        return
    t_min = min(valid_orig.min(), valid_synth.min()) if not valid_synth.empty else valid_orig.min()
    t_max = max(valid_orig.max(), valid_synth.max()) if not valid_synth.empty else valid_orig.max()
    if t_min == t_max:
        ax.set_title(f"{title}\n(constant)")
        ax.text(0.5, 0.5, "constant value", ha="center", va="center", transform=ax.transAxes)
        return
    bins = pd.date_range(t_min, t_max + pd.Timedelta(seconds=1), periods=41)
    ax.hist(valid_orig, bins=bins, density=True, alpha=0.55, color="steelblue", label="Original")
    ax.hist(valid_synth, bins=bins, density=True, alpha=0.55, color="tomato", label="Synthetic")
    ax.set_ylabel("density")
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(title)
    ax.legend(fontsize=7)


def nan_frac(s: pd.Series) -> float:
    return float(s.isna().mean())


# ─── CASE 1: normal date range, no DP ───────────────────────────────────────

def case_normal_no_dp(pdf: PdfPages) -> None:
    print("\n[Case 1] Normal date range, no DP")
    n = 800
    dates = pd.to_datetime("2019-01-01") + pd.to_timedelta(
        rng.exponential(500, n).astype(int), unit="D"
    )
    df = pd.DataFrame({"date": dates, "value": rng.normal(0, 1, n)})

    m = NPGC(epsilon=None)
    m.fit(df, random_state=1)
    synth = m.sample(800, seed=2)

    meta = m._model_state["marginals"]["date"]
    check("type == datetime", meta["type"] == "datetime")
    check("tz == None", meta["tz"] == "None")
    check("nan_frac == 0", meta["nan_frac"] == 0.0)

    orig_secs = set((dates - pd.Timestamp("1970-01-01")).total_seconds())
    anchors = set(meta["sorted_values"])
    check("no-DP anchors match originals exactly", anchors == orig_secs,
          f"{len(anchors & orig_secs)}/{len(anchors)} overlap")

    check("dtype preserved", str(synth["date"].dtype) == str(df["date"].dtype))
    check("min within range", synth["date"].min() >= df["date"].min())
    check("max within range", synth["date"].max() <= df["date"].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Case 1: Normal date range (no DP)", fontsize=11)
    plot_datetime_marginal(axes[0], df["date"], synth["date"], "Date marginal")
    axes[1].scatter(df["date"], df["value"], alpha=0.3, s=6, color="steelblue", label="Orig")
    axes[1].scatter(synth["date"], synth["value"], alpha=0.3, s=6, color="tomato", label="Synth")
    axes[1].set_title("Copula: date vs value")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 2: DP comparison ε = 1.0 vs ε = 0.1 ──────────────────────────────

def case_dp_comparison(pdf: PdfPages) -> None:
    print("\n[Case 2] DP comparison eps=1.0 vs eps=0.1")
    n = 1000
    dates = pd.to_datetime("2021-06-01") + pd.to_timedelta(
        rng.exponential(200, n).astype(int), unit="D"
    )
    df = pd.DataFrame({"date": dates})

    orig_secs = set((dates - pd.Timestamp("1970-01-01")).total_seconds())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Case 2: DP noise effect on anchors", fontsize=11)

    for i, (eps, label) in enumerate([(None, "No DP"), (1.0, "eps=1.0"), (0.1, "eps=0.1")]):
        m = NPGC(epsilon=eps)
        m.fit(df, random_state=0)
        synth = m.sample(1000, seed=1)
        meta = m._model_state["marginals"]["date"]
        anchors = set(meta["sorted_values"])
        overlap = len(anchors & orig_secs)
        expected_zero = eps is not None
        check(f"{label}: anchors != originals (DP)", not overlap if expected_zero else overlap == len(anchors),
              f"{overlap}/{len(anchors)} match")
        plot_datetime_marginal(axes[i], df["date"], synth["date"], label)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 3: all NaT ────────────────────────────────────────────────────────

def case_all_nat(pdf: PdfPages) -> None:
    print("\n[Case 3] All-NaT datetime column")
    n = 200
    df = pd.DataFrame({
        "date": pd.DatetimeIndex([pd.NaT] * n),
        "x": rng.normal(0, 1, n)
    })

    m = NPGC(epsilon=1.0)
    m.fit(df, random_state=0)
    synth = m.sample(200, seed=1)

    meta = m._model_state["marginals"]["date"]
    check("nan_frac == 1.0 for all-NaT", abs(meta["nan_frac"] - 1.0) < 1e-9)
    check("sorted_values is empty", len(meta["sorted_values"]) == 0)
    check("synth date all NaT", synth["date"].isna().all(),
          f"{synth['date'].isna().mean():.0%} NaT")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Case 3: All-NaT datetime column", fontsize=11)
    plot_datetime_marginal(axes[0], df["date"], synth["date"], "Date marginal (all NaT)")
    axes[1].hist(synth["x"], bins=30, color="tomato", alpha=0.7)
    axes[1].set_title("Other column (numeric) still works")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 4: partial NaT — nan_frac preservation ────────────────────────────

def case_partial_nat(pdf: PdfPages) -> None:
    print("\n[Case 4] Partial NaT — nan_frac preservation")
    n = 600
    target_nan = 0.22

    dates = pd.to_datetime("2020-03-01") + pd.to_timedelta(
        rng.integers(0, 730, n), unit="D"
    )
    dates = dates.to_numpy(dtype="datetime64[ns]").copy()
    mask = rng.random(n) < target_nan
    dates[mask] = np.datetime64("NaT")

    df = pd.DataFrame({"date": pd.DatetimeIndex(dates), "v": rng.exponential(5, n)})

    orig_nf = nan_frac(df["date"])
    print(f"  original nan_frac: {orig_nf:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Case 4: NaT fraction preservation across epsilon values", fontsize=11)

    for i, eps in enumerate([None, 1.0, 0.1]):
        m = NPGC(epsilon=eps)
        m.fit(df, random_state=0)
        synth = m.sample(2000, seed=1)
        synth_nf = nan_frac(synth["date"])
        label = f"eps={eps}" if eps else "No DP"
        tol = 0.05
        check(f"{label} nan_frac ≈ original (tol={tol})",
              abs(synth_nf - orig_nf) < tol,
              f"orig={orig_nf:.3f} synth={synth_nf:.3f}")
        plot_datetime_marginal(axes[i], df["date"], synth["date"], f"{label}\nnan_frac: orig={orig_nf:.2f} synth={synth_nf:.2f}")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 5: constant datetime (all same value) ──────────────────────────────

def case_constant_datetime(pdf: PdfPages) -> None:
    print("\n[Case 5] Constant datetime (all same value)")
    n = 300
    const_ts = pd.Timestamp("2023-07-15 12:00:00")
    df = pd.DataFrame({
        "date": pd.DatetimeIndex([const_ts] * n),
        "v": rng.normal(0, 1, n)
    })

    m = NPGC(epsilon=1.0)
    m.fit(df, random_state=0)
    synth = m.sample(300, seed=1)

    # All synthetic dates should equal the constant (or very close in float seconds)
    valid = synth["date"].dropna()
    check("constant: output is not all NaT", len(valid) > 0)
    check("constant: nan_frac == 0", meta := m._model_state["marginals"]["date"],
          )  # placeholder — evaluate below
    check("constant: nan_frac == 0", abs(m._model_state["marginals"]["date"]["nan_frac"]) < 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Case 5: Constant datetime column", fontsize=11)
    plot_datetime_marginal(axes[0], df["date"], synth["date"], "Date marginal (constant)")
    axes[1].hist(synth["v"], bins=30, color="tomato", alpha=0.7)
    axes[1].set_title("Numeric column (should still vary)")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 6: timezone-aware ──────────────────────────────────────────────────

def case_timezone_aware(pdf: PdfPages) -> None:
    print("\n[Case 6] Timezone-aware datetimes")
    # n=2000 keeps SNR ≈ 10 per bin (2000/100 bins = 20 signal vs noise scale 2)
    n = 2000

    timezones = ["UTC", "US/Eastern", "Europe/Berlin", "Asia/Tokyo"]
    fig, axes = plt.subplots(1, len(timezones), figsize=(16, 4))
    fig.suptitle("Case 6: Timezone-aware datetimes", fontsize=11)

    for ax, tz in zip(axes, timezones):
        ts = pd.date_range("2023-01-01", periods=n, freq="6h", tz=tz)
        jitter = pd.to_timedelta(rng.integers(-120, 120, n), unit="min")
        ts = ts + jitter

        df = pd.DataFrame({"ts": ts, "v": rng.normal(0, 1, n)})

        m = NPGC(epsilon=1.0)
        m.fit(df, random_state=0)
        synth = m.sample(n, seed=1)

        meta = m._model_state["marginals"]["ts"]
        check(f"{tz}: type==datetime", meta["type"] == "datetime")
        check(f"{tz}: tz preserved in meta", meta["tz"] == str(tz))
        check(f"{tz}: synth dtype preserved",
              str(synth["ts"].dtype) == str(df["ts"].dtype),
              f"{synth['ts'].dtype}")
        check(f"{tz}: nan_frac == 0", abs(meta["nan_frac"]) < 1e-9)
        check(f"{tz}: min within range", synth["ts"].dropna().min() >= df["ts"].min())
        check(f"{tz}: max within range", synth["ts"].dropna().max() <= df["ts"].max())

        plot_datetime_marginal(ax, df["ts"], synth["ts"], tz, date_fmt="%m-%d")

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 7: intra-day timestamps — temporal resolution check ────────────────

def case_intraday(pdf: PdfPages) -> None:
    print("\n[Case 7] Intra-day timestamps — temporal resolution")
    n = 1000
    base = pd.Timestamp("2024-01-15")
    # Events clustered in business hours (9-17)
    hour_offsets = np.clip(rng.normal(13, 2, n), 8, 18)
    minute_offsets = rng.integers(0, 60, n)
    timestamps = pd.DatetimeIndex([
        base + pd.Timedelta(hours=float(h), minutes=int(m))
        for h, m in zip(hour_offsets, minute_offsets)
    ])
    df = pd.DataFrame({"ts": timestamps, "load": rng.exponential(10, n)})

    bin_width = (df["ts"].max() - df["ts"].min()) / 100
    print(f"  auto bin width: {bin_width}")
    check("bin_width < 1h for single-day data", bin_width < pd.Timedelta(hours=1),
          str(bin_width))

    m = NPGC(epsilon=1.0)
    m.fit(df, random_state=0)
    synth = m.sample(1000, seed=1)

    orig_hours = df["ts"].dt.hour + df["ts"].dt.minute / 60
    synth_hours = synth["ts"].dt.hour + synth["ts"].dt.minute / 60

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Case 7: Intra-day timestamps", fontsize=11)
    axes[0].hist(orig_hours, bins=30, alpha=0.6, color="steelblue", label="Original")
    axes[0].hist(synth_hours, bins=30, alpha=0.6, color="tomato", label="Synthetic (eps=1)")
    axes[0].set_xlabel("Hour of day")
    axes[0].set_title("Intra-day hour distribution")
    axes[0].legend()
    axes[1].scatter(df["ts"], df["load"], alpha=0.3, s=6, color="steelblue", label="Orig")
    axes[1].scatter(synth["ts"], synth["load"], alpha=0.3, s=6, color="tomato", label="Synth")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Time vs Load")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 8: enforce_min_max_values=False ────────────────────────────────────

def case_no_enforce(pdf: PdfPages) -> None:
    print("\n[Case 8] enforce_min_max_values=False — allow extrapolation")
    n = 400
    dates = pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 365, n), unit="D"
    )
    df = pd.DataFrame({"date": dates})

    m_clamp = NPGC(epsilon=None, enforce_min_max_values=True)
    m_clamp.fit(df, random_state=0)
    synth_clamp = m_clamp.sample(1000, seed=1)

    m_free = NPGC(epsilon=None, enforce_min_max_values=False)
    m_free.fit(df, random_state=0)
    synth_free = m_free.sample(1000, seed=1)

    clamp_ok = synth_clamp["date"].dropna().min() >= df["date"].min() and \
               synth_clamp["date"].dropna().max() <= df["date"].max()
    check("enforce=True: synth within training range", clamp_ok)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Case 8: enforce_min_max_values", fontsize=11)
    for ax, synth, label in [(axes[0], synth_clamp, "enforce=True (clamped)"),
                              (axes[1], synth_free, "enforce=False (extrapolation allowed)")]:
        plot_datetime_marginal(ax, df["date"], synth["date"], label)
        ax.axvline(df["date"].min(), color="green", linestyle="--", linewidth=1, label="train min")
        ax.axvline(df["date"].max(), color="green", linestyle="--", linewidth=1, label="train max")
        ax.legend(fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── CASE 9: mixed DataFrame — all column types together ─────────────────────

def case_mixed(pdf: PdfPages) -> None:
    print("\n[Case 9] Mixed DataFrame — datetime + integer + continuous + categorical")
    n = 1000
    signup = pd.to_datetime("2020-01-01") + pd.to_timedelta(
        rng.exponential(500, n).astype(int), unit="D"
    )
    days = (signup - signup.min()).days.values
    revenue = (50 + 0.4 * days + rng.normal(0, 60, n)).clip(0)
    age = rng.integers(18, 70, n)
    plan = rng.choice(["free", "pro", "enterprise"], n, p=[0.6, 0.3, 0.1])

    df = pd.DataFrame({"signup_date": signup, "age": age, "plan": plan, "revenue": revenue})

    m = NPGC(epsilon=1.0)
    m.fit(df, random_state=0)
    synth = m.sample(1000, seed=1)

    # dtype checks
    check("datetime dtype preserved in mixed", str(synth["signup_date"].dtype) == str(df["signup_date"].dtype))
    check("integer dtype preserved", "int" in str(synth["age"].dtype).lower())
    check("categorical preserved", synth["plan"].isin(["free", "pro", "enterprise"]).all())

    # Correlation preservation
    orig_corr = df["signup_date"].astype(np.int64).corr(df["revenue"])
    synth_corr = synth["signup_date"].astype(np.int64).corr(synth["revenue"])
    check("correlation preserved (|diff| < 0.15)",
          abs(orig_corr - synth_corr) < 0.15,
          f"orig={orig_corr:.3f} synth={synth_corr:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Case 9: Mixed DataFrame", fontsize=11)

    bins = pd.date_range(signup.min(), signup.max() + pd.Timedelta(days=1), periods=40)
    axes[0, 0].hist(signup, bins=bins, alpha=0.55, color="steelblue", label="Original")
    axes[0, 0].hist(synth["signup_date"], bins=bins, alpha=0.55, color="tomato", label="Synthetic")
    axes[0, 0].set_title("signup_date marginal")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].tick_params(axis="x", rotation=30)
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].scatter(df["signup_date"], df["revenue"], alpha=0.3, s=5, color="steelblue", label="Orig")
    axes[0, 1].scatter(synth["signup_date"], synth["revenue"], alpha=0.3, s=5, color="tomato", label="Synth")
    axes[0, 1].set_title(f"Date vs Revenue\norig r={orig_corr:.2f}, synth r={synth_corr:.2f}")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 1].tick_params(axis="x", rotation=30)
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].hist(df["age"], bins=20, alpha=0.55, color="steelblue", label="Orig")
    axes[1, 0].hist(synth["age"], bins=20, alpha=0.55, color="tomato", label="Synth")
    axes[1, 0].set_title("age marginal")
    axes[1, 0].legend(fontsize=7)

    orig_plan = df["plan"].value_counts(normalize=True).sort_index()
    synth_plan = synth["plan"].value_counts(normalize=True).sort_index()
    x = np.arange(len(orig_plan))
    axes[1, 1].bar(x - 0.2, orig_plan.values, 0.4, label="Orig", color="steelblue", alpha=0.7)
    axes[1, 1].bar(x + 0.2, synth_plan.reindex(orig_plan.index, fill_value=0).values, 0.4,
                   label="Synth", color="tomato", alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(orig_plan.index)
    axes[1, 1].set_title("plan distribution")
    axes[1, 1].legend(fontsize=7)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Writing report to {PDF_PATH}")
    with PdfPages(PDF_PATH) as pdf:
        case_normal_no_dp(pdf)
        case_dp_comparison(pdf)
        case_all_nat(pdf)
        case_partial_nat(pdf)
        case_constant_datetime(pdf)
        case_timezone_aware(pdf)
        case_intraday(pdf)
        case_no_enforce(pdf)
        case_mixed(pdf)

    print(f"\n{'='*50}")
    if failures:
        print(f"FAILED ({len(failures)} checks):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All checks PASSED")
        print(f"Report: {PDF_PATH}")


if __name__ == "__main__":
    main()
