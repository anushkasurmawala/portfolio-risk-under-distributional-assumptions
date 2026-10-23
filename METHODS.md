# Mathematical and statistical specification

## Portfolio and loss convention

Six source return columns are decimal daily total returns after dividing the
provider's percentage-point values by 100. Weights are fixed at 1/6 per sleeve
at the start of each trading day. The one-day portfolio return is

r_p,t = sum_j w_j r_j,t,    L_t = −r_p,t.

Positive L is a loss fraction. Multiplication by a hypothetical notional V
converts loss fractions to dollar losses. The dollar examples are not capital
requirements. No risk-free return is deducted.

For ten-day simulation, L_10 = −sum_{t=1}^{10} r_p,t. This is an additive,
constant-notional P&L experiment, not a compounded wealth calculation. Daily
cross-sleeve dependence is already reflected in the aggregated portfolio
series; only the separate covariance sensitivity calculation changes it.

## VaR and ES

For a loss distribution F and confidence alpha:

VaR_alpha = inf{x : F(x) >= alpha}.
ES_alpha = integral_alpha^1 VaR_u du / (1−alpha).

For continuous distributions this is the mean loss conditional on being in
the worst (1−alpha) tail. The quantile-integral definition also handles discrete
empirical distributions and their probability mass at the threshold.

Gaussian return model r ~ N(mu, sigma²):

VaR_alpha = −mu + sigma z_alpha.
ES_alpha = −mu + sigma phi(z_alpha)/(1−alpha).

Student t return model r = location + scale T_df:

VaR_alpha = −location + scale q_alpha.
ES_alpha = −location + scale (df + q_alpha²) t_df(q_alpha)
           / ((df−1)(1−alpha)).

z_alpha and q_alpha are standardized Gaussian and t quantiles. phi and t_df
are standardized densities. The t ES formula needs df > 1; this experiment
imposes df >= 2.05 for finite variance as well. Student t scale is not SD:
SD = scale sqrt(df/(df−2)).

For empirical ES, sort n losses from largest to smallest. Let m=n(1−alpha),
k=floor(m), f=m−k. Sum the k largest losses and f times the next loss, then
divide by m. VaR uses the inverted empirical CDF. All figures use percentage
units; all numerical CSV risk values use decimal fractions unless a column
explicitly identifies a dollar notional.

## Estimation and constraints

Gaussian maximum likelihood uses the sample mean and SD with ddof=0. Student
t maximum likelihood minimizes negative log likelihood after normalizing
returns by the Gaussian mean/SD. Optimization bounds in normalized units are
2.05 <= df <= 200, −2 <= location <= 2, and 0.05 <= scale <= 5. Two initial
degrees of freedom (4 and 10) are tried with L-BFGS-B; the lowest successful
objective is retained. The bounds and initializations are modelling choices,
not data-driven claims of optimality. Failure halts execution.

The fitted models need not share moments. In the separate fixed-moment
sensitivity, location equals Gaussian mu and t scale equals
sigma sqrt((df−2)/df), for df 3, 5, 10 and 30.

## Forecast chronology

Frozen estimates use 2010–2019 only. Rolling estimates use the previous 750
trading observations at each month's first trading day and remain fixed for
that month. Every fit end date precedes every associated forecast date.
Out-of-sample evaluation covers 2020–2024. Parameters are not selected using
test outcomes. Later test observations become historical training observations
for subsequent rolling refits, as in a sequential forecasting experiment.

The provider's historical snapshot may have been revised after the evaluation
period, so chronology of return usage does not imply point-in-time data-vintage
availability. Comparisons change recency, window length and fitted parameters
together, and do not identify one isolated causal explanation.

## Bootstrap and simulation uncertainty

Parameter intervals use 250 circular moving-block bootstrap samples, block
length ten trading days, each trimmed to n training observations. All three
risk models are refitted/recomputed on every sample. Percentile intervals
are not simultaneous, and condition on this resampling design and model family.

Monte Carlo uses 200,000 paths, divided into twenty independent simulation
batches. SE for each risk estimate is the sample SD of the twenty batch
estimates divided by sqrt(20). Approximate intervals are estimate +/- 1.96 SE.
They measure simulation error, not parameter or future-return uncertainty.
The block simulation samples one consecutive circular ten-day training block
per path; it preserves within-path observed dependence but cannot recreate
unseen crises. All parametric return models have unbounded support.

## Coverage and clustering tests

Exceedance indicator I_t = 1{L_t > VaR_t}. At confidence alpha the nominal
probability is p=1−alpha. The exact binomial test and Clopper–Pearson interval
use the observed count k over n forecast days, under an independent-hit model.

Kupiec LR = 2(log likelihood under k/n − log likelihood under p), compared
nominally with chi-square(1).

The first-order independence LR compares a two-state Markov hit model with
a constant hit probability using observed transitions n00,n01,n10,n11. It
is nominally compared with chi-square(1). Zero transition counts are handled
with the limit convention 0 log 0 = 0. Sparse transitions and dependence can
make asymptotic or binomial interpretations unreliable. No adjustment for
multiple tests is made. Failure to reject does not prove calibration.

Quantile loss is (alpha − 1{L < VaR}) (L−VaR). Lower average scores are better
when comparing the same confidence level on the same observations.

Average realized tail loss and predicted ES on exceedance days are descriptive
severity diagnostics, not a formal joint VaR/ES calibration test. A formal ES
backtest, conditional-volatility model and ten-day horizon validation are not
implemented in this project.
