"""Run the complete distribution-sensitivity experiment from raw public data."""

from pathlib import Path
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import platform
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from risk import (
    fit_distributions,
    model_risk,
    empirical_risk,
    circular_block_indices,
    backtest_statistics,
)

ROOT = Path(__file__).resolve().parents[1]
URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/6_Portfolios_2x3_daily_CSV.zip"
RAW_NAME = "6_Portfolios_2x3_Daily.csv"
SEED = 2026
WEIGHTS = np.repeat(1 / 6, 6)
ALPHAS = [0.95, 0.99]
MODELS = ["Gaussian", "Student t", "Historical"]
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2024-12-31"
WINDOW = 750


def load_returns():
    raw = ROOT / "data/raw" / RAW_NAME
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        payload = urllib.request.urlopen(URL, timeout=90).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            name = next(n for n in z.namelist() if n.lower() == RAW_NAME.lower())
            raw.write_bytes(z.read(name))
    lines = raw.read_text().splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if "Average Value Weighted Returns -- Daily" in line
    )
    header = lines[start + 1]
    records = []
    for line in lines[start + 2 :]:
        first = line.split(",")[0].strip()
        if not (len(first) == 8 and first.isdigit()):
            break
        records.append(line)
    frame = pd.read_csv(io.StringIO("\n".join([header] + records)), index_col=0)
    frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d")
    frame.index.name = "date"
    frame.columns = frame.columns.str.strip()
    assert (
        frame.shape[1] == 6
        and frame.index.is_unique
        and frame.index.is_monotonic_increasing
    )
    selected = frame.loc[TRAIN_START:TEST_END].copy()
    missing = selected.isin([-99.99, -999]).any(axis=1)
    removed = int(missing.sum())
    selected = selected.loc[~missing] / 100  # source reports percentage points
    assert np.isfinite(selected.to_numpy()).all() and (selected.to_numpy() > -1).all()
    selected["portfolio"] = selected.to_numpy() @ WEIGHTS
    return selected, dict(
        raw_daily_rows=len(frame),
        selected_complete_rows=len(selected),
        missing_rows_removed=removed,
        source_header=lines[0],
    )


def plot_eda(data, train, fit, out):
    r = data.portfolio
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(r.index, 100 * r, color="#245b78", linewidth=0.6)
    axes[0].axvline(pd.Timestamp(TEST_START), color="gray", linestyle="--")
    axes[0].set(
        ylabel="Daily return (%)",
        title="Equal-allocation six-sleeve equity portfolio: daily returns",
    )
    axes[1].plot(r.index, 100 * r.rolling(63).std(ddof=0), color="#245b78")
    axes[1].set(
        xlabel="Date",
        ylabel="Daily standard deviation (%)",
        title="Trailing 63-trading-day volatility (not annualized)",
    )
    fig.tight_layout()
    fig.savefig(out / "return_history.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.linspace(-0.05, 0.05, 500)
    axes[0, 0].hist(
        100 * train, bins=60, density=True, alpha=0.35, label="Training returns"
    )
    axes[0, 0].plot(
        100 * x, stats.norm.pdf(x, fit.mean, fit.sigma) / 100, label="Gaussian"
    )
    axes[0, 0].plot(
        100 * x,
        stats.t.pdf(x, fit.t_df, fit.t_location, fit.t_scale) / 100,
        label="Student t",
    )
    axes[0, 0].set(
        xlabel="Daily return (%)",
        ylabel="Density per percentage point",
        title="2010–2019 return distribution",
    )
    axes[0, 0].legend()
    theoretical, ordered = stats.probplot(
        (train - fit.mean) / fit.sigma, dist="norm", fit=False
    )
    axes[0, 1].scatter(theoretical, ordered, s=6, alpha=0.5)
    axes[0, 1].plot([-4, 4], [-4, 4], color="gray", linestyle="--")
    axes[0, 1].set(
        xlabel="Theoretical standard Gaussian quantile",
        ylabel="Observed standardized return",
        title="Gaussian Q–Q plot: training returns",
    )
    acf = [pd.Series(np.abs(train)).autocorr(lag) for lag in range(1, 21)]
    axes[1, 0].bar(range(1, 21), acf, color="#245b78")
    axes[1, 0].set(
        xlabel="Lag (trading days)",
        ylabel="Sample autocorrelation",
        title="Absolute-return autocorrelation: clustering diagnostic",
    )
    corr = data.loc[TRAIN_START:TRAIN_END].iloc[:, :6].corr()
    image = axes[1, 1].imshow(corr, vmin=0, vmax=1, cmap="Blues")
    axes[1, 1].set_xticks(range(6), corr.columns, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_yticks(range(6), corr.columns, fontsize=8)
    axes[1, 1].set(
        title="Training-period sleeve return correlation",
        xlabel="Research portfolio",
        ylabel="Research portfolio",
    )
    fig.colorbar(image, ax=axes[1, 1], label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(out / "distribution_diagnostics.png", dpi=170)
    plt.close(fig)
    losses = -train
    thresholds = np.linspace(0, 0.07, 300)
    empirical = np.array([(losses > v).mean() for v in thresholds])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        100 * thresholds,
        stats.norm.sf(thresholds, -fit.mean, fit.sigma),
        label="Gaussian",
    )
    ax.plot(
        100 * thresholds,
        stats.t.sf(thresholds, fit.t_df, -fit.t_location, fit.t_scale),
        label="Student t",
    )
    nonzero = empirical > 0
    ax.step(
        100 * thresholds[nonzero],
        empirical[nonzero],
        where="post",
        label="Empirical training distribution",
    )
    ax.set(
        yscale="log",
        ylim=(1e-5, 1),
        xlabel="One-day loss threshold (%)",
        ylabel="Probability of exceeding threshold (log scale)",
        title="Tail sensitivity: 2010–2019 fitted loss distributions",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tail_comparison.png", dpi=170)
    plt.close(fig)


def parameter_uncertainty(train, reps, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        ix = circular_block_indices(len(train), len(train), 10, rng)
        sample = train[ix]
        fit = fit_distributions(sample)
        for model in MODELS:
            for alpha in ALPHAS:
                var, es = model_risk(model, fit, sample, alpha)
                rows.append(
                    dict(replicate=rep, model=model, alpha=alpha, var=var, es=es)
                )
    bootstrap = pd.DataFrame(rows)
    intervals = []
    for (model, alpha), group in bootstrap.groupby(["model", "alpha"]):
        intervals.append(
            dict(
                model=model,
                alpha=alpha,
                var_ci_lo=group["var"].quantile(0.025),
                var_ci_hi=group["var"].quantile(0.975),
                es_ci_lo=group.es.quantile(0.025),
                es_ci_hi=group.es.quantile(0.975),
            )
        )
    return bootstrap, pd.DataFrame(intervals)


def simulate(train, fit, paths, seed):
    """Constant-notional additive P&L, not compound wealth. 20 MC batches."""
    rng = np.random.default_rng(seed)
    rows = []
    for horizon in [1, 10]:
        draws = {
            "Gaussian iid": rng.normal(fit.mean, fit.sigma, size=(paths, horizon)),
            "Student t iid": fit.t_location
            + fit.t_scale * rng.standard_t(fit.t_df, size=(paths, horizon)),
            "Historical iid": train[rng.integers(0, len(train), size=(paths, horizon))],
        }
        if horizon == 10:
            starts = rng.integers(0, len(train), paths)
            draws["Historical 10-day blocks"] = train[
                (starts[:, None] + np.arange(horizon)) % len(train)
            ]
        for model, returns in draws.items():
            losses = -returns.sum(axis=1)
            for alpha in ALPHAS:
                var, es = empirical_risk(losses, alpha)
                batches = [empirical_risk(b, alpha) for b in np.array_split(losses, 20)]
                var_se = np.std([b[0] for b in batches], ddof=1) / np.sqrt(20)
                es_se = np.std([b[1] for b in batches], ddof=1) / np.sqrt(20)
                rows.append(
                    dict(
                        model=model,
                        horizon_days=horizon,
                        alpha=alpha,
                        paths=paths,
                        var=var,
                        es=es,
                        var_mc_se=var_se,
                        es_mc_se=es_se,
                        var_mc_lo=var - 1.96 * var_se,
                        var_mc_hi=var + 1.96 * var_se,
                        es_mc_lo=es - 1.96 * es_se,
                        es_mc_hi=es + 1.96 * es_se,
                    )
                )
    return pd.DataFrame(rows)


def backtest(data, training_fit):
    """Frozen pre-2020 fits vs 750-day fits updated at each month's first day."""
    history = data.portfolio
    train = history.loc[TRAIN_START:TRAIN_END].to_numpy()
    test = history.loc[TEST_START:TEST_END]
    rows = []
    params = []
    current_month = None
    for date, ret in test.items():
        month = date.to_period("M")
        if month != current_month:
            window = history.loc[history.index < date].iloc[-WINDOW:]
            assert len(window) == WINDOW and window.index.max() < date
            rolling_fit = fit_distributions(window.to_numpy())
            current_month = month
            params.append(
                dict(
                    refit_date=date,
                    fit_start=window.index.min(),
                    fit_end=window.index.max(),
                    observations=len(window),
                    **asdict(rolling_fit),
                )
            )
        for scheme, fit, sample, start, end in [
            (
                "Frozen 2010–2019",
                training_fit,
                train,
                history.loc[TRAIN_START:TRAIN_END].index.min(),
                history.loc[TRAIN_START:TRAIN_END].index.max(),
            ),
            (
                "Rolling 750 days",
                rolling_fit,
                window.to_numpy(),
                window.index.min(),
                window.index.max(),
            ),
        ]:
            for model in MODELS:
                for alpha in ALPHAS:
                    var, es = model_risk(model, fit, sample, alpha)
                    rows.append(
                        dict(
                            date=date,
                            scheme=scheme,
                            model=model,
                            alpha=alpha,
                            loss=-ret,
                            var=var,
                            es=es,
                            exceeded=bool(-ret > var),
                            fit_start=start,
                            fit_end=end,
                        )
                    )
    forecasts = pd.DataFrame(rows)
    summary = []
    for (scheme, model, alpha), group in forecasts.groupby(
        ["scheme", "model", "alpha"]
    ):
        for period, subset in [("2020–2024", group)] + [
            (str(year), group[group.date.dt.year == year]) for year in range(2020, 2025)
        ]:
            summary.append(
                dict(
                    scheme=scheme,
                    model=model,
                    alpha=alpha,
                    period=period,
                    **backtest_statistics(subset.loss, subset["var"], subset.es, alpha),
                )
            )
    return forecasts, pd.DataFrame(summary), pd.DataFrame(params)


def main(bootstrap=250, paths=200000):
    out = ROOT / "reports"
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data, audit = load_returns()
    data.to_csv(out / "daily_returns.csv")
    train = data.loc[TRAIN_START:TRAIN_END, "portfolio"].to_numpy()
    fit = fit_distributions(train)
    eda = []
    for name, series in [
        ("Training 2010–2019", data.loc[TRAIN_START:TRAIN_END, "portfolio"]),
        ("Test 2020–2024", data.loc[TEST_START:TEST_END, "portfolio"]),
    ]:
        eda.append(
            dict(
                period=name,
                n=len(series),
                mean=series.mean(),
                std=series.std(ddof=0),
                skew=stats.skew(series, bias=False),
                excess_kurtosis=stats.kurtosis(series, bias=False),
                minimum=series.min(),
                maximum=series.max(),
                return_acf1=series.autocorr(1),
                absolute_return_acf1=series.abs().autocorr(1),
            )
        )
    pd.DataFrame(eda).to_csv(out / "eda_summary.csv", index=False)
    data.loc[TRAIN_START:TRAIN_END].iloc[:, :6].corr().to_csv(
        out / "training_correlations.csv"
    )
    boot, intervals = parameter_uncertainty(train, bootstrap, SEED)
    boot.to_csv(out / "bootstrap_estimates.csv", index=False)
    rows = []
    for model in MODELS:
        for alpha in ALPHAS:
            var, es = model_risk(model, fit, train, alpha)
            rows.append(
                dict(
                    model=model,
                    alpha=alpha,
                    var=var,
                    es=es,
                    var_per_100k=100000 * var,
                    es_per_100k=100000 * es,
                )
            )
    estimates = pd.DataFrame(rows).merge(intervals, on=["model", "alpha"])
    estimates.to_csv(out / "one_day_risk.csv", index=False)
    (out / "training_fit.json").write_text(json.dumps(asdict(fit), indent=2))
    sensitivity = []
    for df in [3, 5, 10, 30]:
        matched_scale = fit.sigma * np.sqrt((df - 2) / df)
        for alpha in ALPHAS:
            z = stats.t.ppf(alpha, df)
            sensitivity.append(
                dict(
                    df=df,
                    alpha=alpha,
                    matched_mean=fit.mean,
                    matched_sd=fit.sigma,
                    scale=matched_scale,
                    var=-fit.mean + matched_scale * z,
                    es=-fit.mean
                    + matched_scale
                    * (df + z * z)
                    / (df - 1)
                    * stats.t.pdf(z, df)
                    / (1 - alpha),
                )
            )
    pd.DataFrame(sensitivity).to_csv(out / "tail_df_sensitivity.csv", index=False)
    covariance = np.cov(
        data.loc[TRAIN_START:TRAIN_END].iloc[:, :6].to_numpy(), rowvar=False, ddof=0
    )
    full_sigma = float(np.sqrt(WEIGHTS @ covariance @ WEIGHTS))
    independent_sigma = float(np.sqrt(np.sum(WEIGHTS**2 * np.diag(covariance))))
    dependence = []
    for assumption, sigma in [
        ("Observed covariance", full_sigma),
        ("Zero cross-sleeve correlation", independent_sigma),
    ]:
        for alpha in ALPHAS:
            z = stats.norm.ppf(alpha)
            dependence.append(
                dict(
                    assumption=assumption,
                    alpha=alpha,
                    sigma=sigma,
                    var=-fit.mean + sigma * z,
                    es=-fit.mean + sigma * stats.norm.pdf(z) / (1 - alpha),
                )
            )
    pd.DataFrame(dependence).to_csv(out / "correlation_sensitivity.csv", index=False)
    mc = simulate(train, fit, paths, SEED + 1)
    mc.to_csv(out / "monte_carlo_risk.csv", index=False)
    forecasts, summary, params = backtest(data, fit)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    summary.to_csv(out / "backtest_summary.csv", index=False)
    params.to_csv(out / "rolling_fit_parameters.csv", index=False)
    plot_eda(data, train, fit, figures)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, (metric, label) in enumerate(
        [("var", "Value at Risk"), ("es", "Expected Shortfall")]
    ):
        for i, alpha in enumerate(ALPHAS):
            points = estimates[estimates.alpha == alpha].set_index("model").loc[MODELS]
            pos = np.arange(3) + (i - 0.5) * 0.3
            axes[j].bar(pos, 100 * points[metric], width=0.3, label=f"{alpha:.0%}")
            axes[j].vlines(
                pos,
                100 * points[metric + "_ci_lo"],
                100 * points[metric + "_ci_hi"],
                color="black",
                linewidth=1,
            )
        axes[j].set_xticks(range(3), MODELS)
        axes[j].set(
            ylabel="One-day loss (% of notional)",
            xlabel="Distribution assumption",
            title=label + " with block-bootstrap intervals",
        )
        axes[j].legend()
    fig.tight_layout()
    fig.savefig(figures / "one_day_risk.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, scheme in zip(axes, ["Frozen 2010–2019", "Rolling 750 days"]):
        subset = forecasts[(forecasts.scheme == scheme) & (forecasts.alpha == 0.99)]
        realized = subset[subset.model == "Gaussian"]
        ax.plot(
            realized.date,
            100 * realized.loss,
            color="gray",
            alpha=0.5,
            linewidth=0.6,
            label="Observed portfolio loss",
        )
        for model in MODELS:
            p = subset[subset.model == model]
            ax.plot(p.date, 100 * p["var"], linewidth=1, label=model + " 99% VaR")
        ax.set(ylabel="One-day loss (% of notional)", title=scheme)
        ax.legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Forecast date")
    fig.tight_layout()
    fig.savefig(figures / "var_backtest.png", dpi=170)
    plt.close(fig)
    metadata = dict(
        run_time_utc=datetime.now(timezone.utc).isoformat(),
        seed=SEED,
        bootstrap_replicates=bootstrap,
        bootstrap_block_length=10,
        monte_carlo_paths=paths,
        rolling_window=WINDOW,
        rolling_refits=len(params),
        train_dates=[TRAIN_START, TRAIN_END],
        test_dates=[TEST_START, TEST_END],
        weights=WEIGHTS.tolist(),
        audit=audit,
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        scipy=scipy.__version__,
        matplotlib=matplotlib.__version__,
        raw_sha256=hashlib.sha256(
            (ROOT / "data/raw" / RAW_NAME).read_bytes()
        ).hexdigest(),
        source_url=URL,
    )
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("Training fit:", asdict(fit))
    print(estimates.to_string(index=False))
    print(
        summary[summary.period == "2020–2024"][
            [
                "scheme",
                "model",
                "alpha",
                "n",
                "exceedances",
                "expected_exceedances",
                "exceedance_rate",
                "binomial_pvalue",
                "independence_pvalue",
            ]
        ].to_string(index=False)
    )
    print("Saved analysis to", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=250)
    parser.add_argument("--paths", type=int, default=200000)
    args = parser.parse_args()
    if args.bootstrap < 100 or args.paths < 20000 or args.paths % 20:
        parser.error("Require bootstrap >=100 and paths >=20000 divisible by 20")
    main(args.bootstrap, args.paths)
