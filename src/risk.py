"""Distribution fitting, coherent empirical tail means, and risk backtests.

Returns and losses are decimal fractions. Positive loss means losing money.
"""

from dataclasses import dataclass
import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class DistributionFit:
    mean: float
    sigma: float
    t_df: float
    t_location: float
    t_scale: float
    t_optimizer_success: bool


def fit_distributions(returns):
    """Fit Gaussian MLE and bounded Student t MLE; normalize for conditioning."""
    x = np.asarray(returns, dtype=float)
    if len(x) < 100 or not np.isfinite(x).all():
        raise ValueError("At least 100 finite return observations are required.")
    mean, sigma = float(x.mean()), float(x.std(ddof=0))
    if sigma <= 0:
        raise ValueError("Returns must have positive variance.")
    z = (x - mean) / sigma

    def nll(parameters):
        df, location, log_scale = parameters
        return -np.sum(stats.t.logpdf(z, df, loc=location, scale=np.exp(log_scale)))

    solutions = [
        optimize.minimize(
            nll,
            [df, 0, np.log(0.8)],
            method="L-BFGS-B",
            bounds=[(2.05, 200), (-2, 2), (np.log(0.05), np.log(5))],
        )
        for df in [4, 10]
    ]
    successful = [s for s in solutions if s.success and np.isfinite(s.fun)]
    if not successful:
        raise RuntimeError(
            f"Student t optimization failed: {[s.message for s in solutions]}"
        )
    best = min(successful, key=lambda s: s.fun)
    df, location, log_scale = best.x
    return DistributionFit(
        mean,
        sigma,
        float(df),
        mean + sigma * float(location),
        sigma * float(np.exp(log_scale)),
        True,
    )


def empirical_risk(losses, alpha):
    """Empirical VaR and quantile-integral ES, including a fractional tail atom.

    This defines ES for discrete empirical distributions without relying on
    the potentially wrong mean(loss > VaR), which discards the threshold atom.
    """
    x = np.asarray(losses, dtype=float)
    if not 0 < alpha < 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Require finite nonempty losses and alpha in (0,1).")
    var = float(np.quantile(x, alpha, method="inverted_cdf"))
    descending = np.sort(x)[::-1]
    tail_mass = len(x) * (1 - alpha)
    complete = int(np.floor(tail_mass))
    fraction = tail_mass - complete
    total = descending[:complete].sum()
    if fraction > 0:
        total += fraction * descending[complete]
    return var, float(total / tail_mass)


def model_risk(model, fit, returns, alpha):
    """One-day loss VaR/ES; exact parametric formula or empirical quantiles."""
    if model == "Historical":
        return empirical_risk(-np.asarray(returns), alpha)
    if model == "Gaussian":
        z = stats.norm.ppf(alpha)
        return (
            -fit.mean + fit.sigma * z,
            -fit.mean + fit.sigma * stats.norm.pdf(z) / (1 - alpha),
        )
    if model == "Student t":
        z = stats.t.ppf(alpha, fit.t_df)
        return (
            -fit.t_location + fit.t_scale * z,
            -fit.t_location
            + fit.t_scale
            * (fit.t_df + z * z)
            / (fit.t_df - 1)
            * stats.t.pdf(z, fit.t_df)
            / (1 - alpha),
        )
    raise ValueError(model)


def circular_block_indices(n, size, block_length, rng):
    """Sample circular consecutive blocks, trimming to the requested size."""
    starts = rng.integers(0, n, int(np.ceil(size / block_length)))
    return ((starts[:, None] + np.arange(block_length)) % n).ravel()[:size]


def bernoulli_log_likelihood(successes, trials, probability):
    return float(
        special.xlogy(successes, probability)
        + special.xlogy(trials - successes, 1 - probability)
    )


def backtest_statistics(losses, var, es, alpha):
    """Coverage and clustering tests; tail-severity is a descriptive ES check."""
    loss, var, es = map(np.asarray, (losses, var, es))
    exceed = loss > var
    k, n = int(exceed.sum()), len(loss)
    expected_probability = 1 - alpha
    fitted_probability = k / n
    likelihood_null = bernoulli_log_likelihood(k, n, expected_probability)
    likelihood_fit = bernoulli_log_likelihood(k, n, fitted_probability)
    lr_coverage = max(0.0, 2 * (likelihood_fit - likelihood_null))
    previous, current = exceed[:-1], exceed[1:]
    n00 = int((~previous & ~current).sum())
    n01 = int((~previous & current).sum())
    n10 = int((previous & ~current).sum())
    n11 = int((previous & current).sum())
    n0, n1 = n00 + n01, n10 + n11
    p0 = n01 / n0 if n0 else 0.0
    p1 = n11 / n1 if n1 else 0.0
    p = (n01 + n11) / (n0 + n1)
    ll_independent = bernoulli_log_likelihood(n01 + n11, n0 + n1, p)
    ll_markov = bernoulli_log_likelihood(n01, n0, p0) + bernoulli_log_likelihood(
        n11, n1, p1
    )
    lr_independence = max(0.0, 2 * (ll_markov - ll_independent))
    lower, upper = stats.binomtest(k, n).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    return dict(
        n=n,
        exceedances=k,
        expected_exceedances=n * expected_probability,
        exceedance_rate=k / n,
        coverage_rate_ci_lo=lower,
        coverage_rate_ci_hi=upper,
        binomial_pvalue=stats.binomtest(k, n, expected_probability).pvalue,
        kupiec_lr=lr_coverage,
        kupiec_pvalue=stats.chi2.sf(lr_coverage, 1),
        independence_lr=lr_independence,
        independence_pvalue=stats.chi2.sf(lr_independence, 1),
        consecutive_exceedances=n11,
        mean_quantile_loss=float(np.mean((alpha - (loss < var)) * (loss - var))),
        mean_loss_on_exceedance=float(loss[exceed].mean()) if k else np.nan,
        mean_es_on_exceedance=float(es[exceed].mean()) if k else np.nan,
    )
