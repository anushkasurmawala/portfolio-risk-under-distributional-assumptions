"""Verify chronology, unit conversion, coverage arithmetic and risk formulas."""

import hashlib
import json
import numpy as np
import pandas as pd
from scipy import integrate, stats
from run_analysis import (
    ROOT,
    RAW_NAME,
    TRAIN_START,
    TRAIN_END,
    TEST_START,
    TEST_END,
    WINDOW,
    WEIGHTS,
    load_returns,
)
from risk import DistributionFit, empirical_risk, model_risk


def main():
    reports = ROOT / "reports"
    metadata = json.loads((reports / "run_metadata.json").read_text())
    assert (
        hashlib.sha256((ROOT / "data/raw" / RAW_NAME).read_bytes()).hexdigest()
        == metadata["raw_sha256"]
    )
    data, _ = load_returns()
    saved = pd.read_csv(
        reports / "daily_returns.csv", index_col="date", parse_dates=True
    )
    assert data.shape == saved.shape and np.allclose(
        data.to_numpy(), saved.to_numpy(), atol=1e-14
    )
    assert np.allclose(saved.iloc[:, :6].to_numpy() @ WEIGHTS, saved.portfolio)
    assert data.index.is_unique and data.index.is_monotonic_increasing
    assert np.isclose(WEIGHTS.sum(), 1)
    train = data.loc[TRAIN_START:TRAIN_END].portfolio.to_numpy()
    test = data.loc[TEST_START:TEST_END].portfolio
    fit = DistributionFit(**json.loads((reports / "training_fit.json").read_text()))
    assert fit.t_df > 2 and fit.t_optimizer_success
    assert np.isclose(train.mean(), fit.mean) and np.isclose(
        train.std(ddof=0), fit.sigma
    )
    # Hand-calculated discrete ES, including a fractional threshold atom.
    assert np.allclose(empirical_risk([1, 2, 3, 4, 5], 0.6), [3, 4.5])
    assert np.allclose(empirical_risk([1, 2, 3, 4, 5], 0.7), [4, 14 / 3])
    # Independently check analytical ES by integrating the loss density.
    estimates = pd.read_csv(reports / "one_day_risk.csv")
    for row in estimates.itertuples():
        var, es = model_risk(row.model, fit, train, row.alpha)
        assert np.allclose([var, es], [row.var, row.es], atol=1e-13)
        assert es >= var and np.isclose(row.var_per_100k, 100000 * var)
        if row.model in ["Gaussian", "Student t"]:
            distribution = (
                stats.norm(loc=-fit.mean, scale=fit.sigma)
                if row.model == "Gaussian"
                else stats.t(fit.t_df, loc=-fit.t_location, scale=fit.t_scale)
            )
            assert np.isclose(distribution.cdf(var), row.alpha)
            numeric = integrate.quad(
                lambda loss: loss * distribution.pdf(loss), var, np.inf, epsabs=1e-11
            )[0] / (1 - row.alpha)
            assert np.isclose(es, numeric, rtol=1e-7, atol=1e-9)
    forecasts = pd.read_csv(
        reports / "forecasts.csv", parse_dates=["date", "fit_start", "fit_end"]
    )
    assert (forecasts.fit_end < forecasts.date).all()
    assert not forecasts.duplicated(["date", "scheme", "model", "alpha"]).any()
    assert len(forecasts) == len(test) * 2 * 3 * 2
    assert np.allclose(forecasts.loss, -test.reindex(forecasts.date).to_numpy())
    assert np.array_equal(
        (forecasts.loss > forecasts["var"]).to_numpy(), forecasts.exceeded.to_numpy()
    )
    params = pd.read_csv(
        reports / "rolling_fit_parameters.csv",
        parse_dates=["refit_date", "fit_start", "fit_end"],
    )
    assert len(params) == 60 and (params.observations == WINDOW).all()
    for row in params.itertuples():
        window = data.loc[data.index < row.refit_date].iloc[-WINDOW:]
        assert window.index.min() == row.fit_start and window.index.max() == row.fit_end
        assert np.isclose(window.portfolio.mean(), row.mean)
        assert np.isclose(window.portfolio.std(ddof=0), row.sigma)
    summary = pd.read_csv(reports / "backtest_summary.csv", dtype={"period": str})
    for row in summary.itertuples():
        subset = forecasts[
            (forecasts.scheme == row.scheme)
            & (forecasts.model == row.model)
            & (forecasts.alpha == row.alpha)
        ]
        if row.period != "2020–2024":
            subset = subset[subset.date.dt.year == int(row.period)]
        k = int((subset.loss > subset["var"]).sum())
        assert k == row.exceedances and len(subset) == row.n
        assert np.isclose(k / len(subset), row.exceedance_rate)
        assert np.isclose(
            stats.binomtest(k, len(subset), 1 - row.alpha).pvalue, row.binomial_pvalue
        )
    # Simulation error is separate from fitted-parameter uncertainty.
    mc = pd.read_csv(reports / "monte_carlo_risk.csv")
    for row in mc.itertuples():
        if row.horizon_days == 1 and row.model in ["Gaussian iid", "Student t iid"]:
            var, es = model_risk(row.model.replace(" iid", ""), fit, train, row.alpha)
        elif row.model == "Gaussian iid":
            h = row.horizon_days
            z = stats.norm.ppf(row.alpha)
            var = -h * fit.mean + np.sqrt(h) * fit.sigma * z
            es = -h * fit.mean + np.sqrt(h) * fit.sigma * stats.norm.pdf(z) / (
                1 - row.alpha
            )
        else:
            continue
        assert abs(row.var - var) < 5 * row.var_mc_se
        assert abs(row.es - es) < 5 * row.es_mc_se
    print(
        "PASS: raw hash, percent-to-decimal data, weights, chronology, 15,096 forecast rows, 72 coverage summaries, hand-calculated empirical ES, numerical ES integrals, and analytical/Monte Carlo agreement."
    )


if __name__ == "__main__":
    main()
