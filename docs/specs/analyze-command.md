# Statistical Factor Importance Analysis

## Goal

Provide a post-run CLI command that estimates which benchmark factors matter most for generation speed (`tg_tokens_per_sec`) using a statistically defensible model that can ship inside the existing Node.js CLI without bundling Python.

## Scope

Add a new `analyze` command that:

- loads benchmark `runs.jsonl` records for a model
- filters to complete successful records
- fits a multi-factor regression model for `tg_tokens_per_sec`
- reports grouped factor importance, uncertainty, and selected interactions
- optionally writes a JSON report for downstream visualization

Out of scope:

- full Bayesian modeling
- high-order interaction search across all factors
- exact causal claims from adaptive sampling

## Constraints

- Must run inside Node.js and bundle with the existing CLI.
- Must not require Python or R.
- Must work on unbalanced designs from the adaptive DOE scan.
- Must tolerate repeated runs per configuration.
- Must remain reasonably transparent and debuggable.

## Model

### Response

- Primary target: `tg_tokens_per_sec`

### Predictors

- Categorical factors: one-hot encoded
- Boolean factors: encoded as binary indicators
- Numeric factors: centered linear term plus quadratic term

### Interactions

- Compute main effects for all factors
- Compute pairwise interactions only for the top 2-4 factors from the main-effects pass

### Estimation

- Ordinary least squares using a design matrix
- Group factor coefficients by source factor
- Measure importance by drop-in model fit when removing a factor group

## Statistical Outputs

For each factor:

- factor name
- effect size estimate
- normalized importance share
- confidence interval from bootstrap
- F statistic and p-value from reduced-vs-full model comparison
- direction summary where meaningful
- nonlinear flag for numeric axes

For the model:

- sample count
- number of unique configs
- R-squared
- adjusted R-squared
- residual variance estimate

For interactions:

- top pairwise interactions by incremental fit contribution

## Robustness

- Use bootstrap resampling over run records to estimate uncertainty
- Report intervals, not only point estimates
- Warn that results are model-based estimates from adaptive sampling
- Prefer effect sizes over p-values in the textual summary

## CLI

Planned command:

```bash
lmstudio-bench analyze --model <model> [--data-dir <path>] [--bootstrap 200] [--top-interactions 3] [--json]
```

Outputs:

- terminal summary table
- optional JSON report written beside model results

## Dependencies

- `ml-matrix` for matrix operations / OLS solving
- `@stdlib/stats-base-dists-f-cdf` for ANOVA-style reduced-model F tests

## Testing

Add tests for:

- grouped factor importance ranking on synthetic data
- quadratic detection for nonlinear numeric effects
- bootstrap interval generation shape / stability
- CLI command registration and JSON output

## Assumptions

- Repeated runs are independent enough to use as repeated observations
- OLS with grouped effects is good enough for ranking factor importance
- Adaptive/hint-biased sampling limits causal interpretation, so output must be phrased as estimated importance, not proof
