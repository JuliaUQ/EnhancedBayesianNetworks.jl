# ==================================================================================================
# Parameter estimators
#
# An estimator answers one question: given the data counts of a single parent configuration, what goes in that row of the CPT? Precise estimators return numbers and yield a `BayesianNetwork`; imprecise ones return `Interval`s and yield a `CredalNetwork`. Everything else — walking the nodes, matching rows, counting — is shared, and does not belong here.  Parameters are validated in the constructors, so an unusable estimator cannot be built and the fitting loop never has to check.
# ==================================================================================================

"""
    ParameterEstimator

Abstract supertype of the estimators accepted by [`learn`](@ref). An estimator fixes *how* the probabilities for one parent configuration are derived from data counts, and with it the kind of network returned:

  - [`PreciseEstimator`](@ref) — one probability per node state, giving a [`BayesianNetwork`](@ref).
  - [`ImpreciseEstimator`](@ref) — an `Interval` per node state, giving a [`CredalNetwork`](@ref).

Every estimator works on **one parent configuration**: one node together with one configuration of its parents' states. Three quantities describe it, and every formula in this module is written in terms of them:

  - `k` — how many states the node has.
  - `cnt` — how many data rows match the parent configuration *and* have the node in the state being estimated.
  - `total` — how many data rows match the parent configuration at all, whatever the node's state.

Available estimators are [`PreciseDirichletModel`](@ref), [`ImpreciseDirichletModel`](@ref) and [`QuesenberryHurst`](@ref).
"""
abstract type ParameterEstimator end

"""
    PreciseEstimator <: ParameterEstimator

Estimators returning a single probability per node state, so that [`learn`](@ref) returns a [`BayesianNetwork`](@ref). See [`PreciseDirichletModel`](@ref).
"""
abstract type PreciseEstimator <: ParameterEstimator end

"""
    ImpreciseEstimator <: ParameterEstimator

Estimators returning an `Interval` per node state, so that [`learn`](@ref) returns a [`CredalNetwork`](@ref). See [`ImpreciseDirichletModel`](@ref) and [`QuesenberryHurst`](@ref).
"""
abstract type ImpreciseEstimator <: ParameterEstimator end

"""
    PreciseDirichletModel(α = 0)

Based on a Dirichlet **prior** that can be written as a **concentration** vector `a = (a₁, …, aₖ)` or in terms of **strength** (s) `s = Σᵢ aᵢ` and **mean** (t) `tᵢ = aᵢ / s`.
The **posterior mean** is then expressed as `E(θᵢ | data) = (cntᵢ + aᵢ) / (total + s) = (cntᵢ + tᵢ * s) / (total + s)`.

See [`ParameterEstimator`](@ref) for `cnt`, `total` and `k`.

The Precise Dirichlet Model evaluates CPT's entries as `(cnt + α) / (total + α * k)`, with `α ≥ 0` a pseudo-count added to each of the `k` states. The pseudo-count `α` is the same in every state, which means the Dirichlet prior is symmetric and its mean is uniform, `tᵢ = 1 / k` for every `i`.
When `α = 0` the posterior falls back to maximum likelihood estimation.

Named special cases, all the same formula:

| `α` | name |
|:--|:--|
| `0` | maximum likelihood, the relative frequency (the default) |
| any | additive, Lidstone or Laplace smoothing |

`α = 0` is unbiased but overconfident on sparse data: a state never observed in a configuration is assigned exactly `0`. Larger `α` tends to smooth away the zero probability, pushing towards a uniform distribution. Use [`ImpreciseDirichletModel`](@ref) to report that uncertainty as an interval rather than smooth it away.

The estimator is due to the actuaries Hardy (1889) and Lidstone (1920), and was recast by Johnson (1932) as the interpolation above. Reference: Lidstone, G. J. (1920), *Note on the general case of the Bayes-Laplace formula for inductive or a posteriori probabilities*, Transactions of the Faculty of Actuaries 8, 182–192.

# Examples
```julia
bn = learn(dag, df, PreciseDirichletModel())      # maximum likelihood
bn = learn(dag, df, PreciseDirichletModel(1))     # Laplace smoothing
order!(bn)
```
"""
struct PreciseDirichletModel <: PreciseEstimator
    α::Float64
    function PreciseDirichletModel(α::Real = 0)
        if !isfinite(α) || α < 0
            error("Invalid PreciseDirichletModel: α must be finite and nonnegative, got $(repr(α))")
        end
        return new(float(α))
    end
end

"""
    ImpreciseDirichletModel(s = 1)

Based on a Dirichlet **prior** that can be written as a **concentration** vector `a = (a₁, …, aₖ)` or in terms of **strength** (s) `s = Σᵢ aᵢ` and **mean** (t) `tᵢ = aᵢ / s`.
The **posterior mean** is then expressed as `E(θᵢ | data) = (cntᵢ + aᵢ) / (total + s) = (cntᵢ + tᵢ * s) / (total + s)`.

See [`ParameterEstimator`](@ref) for `cnt`, `total` and `k`.

The Imprecise Dirichlet Model (IDM) of Walley evaluates CPT's entries as the interval defined by `[cnt / (total + s), (cnt + s) / (total + s)]`, with `s > 0` the prior strength. `s = 1` and `s = 2` are the usual choices; larger `s` gives wider intervals.

The bounds can be read straight off: `s` is a number of extra, imagined observations, and the model is not told which states they belong to. The **lower** bound assumes none of them fall in the evaluated state of the nodes; the **upper** bound assumes all `s` of them do.

Formally, IDM fixes the strength `s` and leaves the mean `tᵢ` free, which is [`PreciseDirichletModel`](@ref) with the prior no longer pinned down. Only `tᵢ` appears in the posterior mean above, and the expression is increasing in it, so letting the mean range over every possible distribution sends `tᵢ = 0` to the lower bound and `tᵢ = 1` to the upper one: the interval is exactly that range. Substituting `tᵢ = 1 / k` recovers `PreciseDirichletModel(s / k)`, since `total + (s / k) * k = total + s`, so the interval always contains the uniform-prior case.

Two properties make it the recommended default for credal learning.

1. It is **coherent by construction**: the lower bounds of a parent configuration sum to `total / (total + s) ≤ 1` and the upper bounds to `(total + k * s) / (total + s) ≥ 1`, so the credal set is never empty.
2. A parent configuration absent from the data gives `[0, 1]` with no special case: with no observations the model says nothing, which is the honest answer and the one [`infer`](@ref) handles via the regular extension.

Reference: Walley, P. (1996), *Inferences from Multinomial Data: Learning About a Bag of Marbles*, Journal of the Royal Statistical Society Series B 58(1), 3–34.

# Examples
```julia
cn = learn(dag, df, ImpreciseDirichletModel())       # s = 1
cn = learn(dag, df, ImpreciseDirichletModel(2))
order!(cn)
```
"""
struct ImpreciseDirichletModel <: ImpreciseEstimator
    s::Float64
    function ImpreciseDirichletModel(s::Real = 1)
        if !isfinite(s) || s <= 0
            error("Invalid ImpreciseDirichletModel: s must be finite and strictly positive, got $(repr(s))")
        end
        return new(float(s))
    end
end

"""
    QuesenberryHurst(confidence = 0.95)

Simultaneous confidence intervals for multinomial proportions: the set of parameters the data does not reject, obtained by inverting Pearson's score test.

Keeping every `θ` with `Σᵢ (cntᵢ - total * θᵢ)^2 / (total * θᵢ) ≤ A` carves an ellipsoid out of the simplex. Projecting it onto one coordinate is a quadratic in `θ`, whose roots are

    θ = [(A + 2 * cnt) ± sqrt(A * (A + 4 * cnt * (total - cnt) / total))] / (2 * (total + A))

with `A = quantile(Chisq(k - 1), confidence)`. The `k - 1` degrees of freedom are what make the coverage **simultaneous** over the `k` node states rather than one state at a time. At `k = 2` this reduces exactly to the Wilson score interval, since `χ²(1, 0.95) = 1.959964^2`.

Unlike [`ImpreciseDirichletModel`](@ref) this is a frequentist confidence region read as a credal set: it carries repeated-sampling coverage, not a behavioural interpretation, so results should say which of the two produced them. Coherence is nonetheless guaranteed whenever `total > 0`, because the projected ellipsoid contains the maximum likelihood estimate, whose probabilities sum to `1`.

Reference: Quesenberry, C. P. and Hurst, D. C. (1964), *Large Sample Simultaneous Confidence Intervals for Multinomial Proportions*, Technometrics 6(2), 191–195.

# Examples
```julia
cn = learn(dag, df, QuesenberryHurst())
cn = learn(dag, df, QuesenberryHurst(0.99))
```
"""
struct QuesenberryHurst <: ImpreciseEstimator
    confidence::Float64
    function QuesenberryHurst(confidence::Real = 0.95)
        if !isfinite(confidence) || !(0 < confidence < 1)
            error("Invalid QuesenberryHurst: confidence must lie strictly between 0 and 1, got $(repr(confidence))")
        end
        return new(float(confidence))
    end
end

# --------------------------------------------------------------------------------------------------
# _estimate(estimator, counts, total) -> one parents' configuration
#
# `counts[i]` is the number of data's rows matching the parent configuration whose node state is the i-th of the node's domain; `total` is the number of rows matching the configuration at all. For complete data `total == sum(counts)`, but the two are passed separately on purpose: under missing data a row can match the parents while leaving the node unobserved. Counts are `Real`, not `Int`, because EM's M-step sums fractional responsibilities.
#
# `k = length(counts)` is read off the row, so anything depending only on `k` — notably the chi-square quantile — is computed once per parents' configuration rather than once per node state.
#
# Each method owns its own degenerate cases; the fitting loop has none.
# --------------------------------------------------------------------------------------------------

function _estimate(estimator::PreciseDirichletModel, counts::AbstractVector{<:Real}, total::Real)
    k = length(counts)
    if k == 1   # node with 1 state only
        return DiscreteProbability[1.0]
    end
    α = estimator.α
    # MLE and parents configuration not present
    if (iszero(total) && iszero(α))
        return DiscreteProbability[1 / k for _ in 1:k]
    end
    # all the other cases are covered, clamp is here only for the em-steps where cnt might be bigger than total
    return DiscreteProbability[clamp((cnt + α) / (total + α * k), 0.0, 1.0) for cnt in counts]
end

function _estimate(estimator::ImpreciseDirichletModel, counts::AbstractVector{<:Real}, total::Real)
    k = length(counts)
    if k == 1   # node with 1 state only
        return DiscreteProbability[1.0]
    end
    s = estimator.s
    # all the other cases are covered, clamp is here only for the em-steps where cnt might be bigger than total
    # total == 0: [0 / s, s / s] = [0, 1]
    return DiscreteProbability[
        Interval(
            clamp(cnt / (total + s), 0.0, 1.0),
            clamp((cnt + s) / (total + s), 0.0, 1.0)
        ) for cnt in counts
    ]
end

function _estimate(estimator::QuesenberryHurst, counts::AbstractVector{<:Real}, total::Real)
    k = length(counts)
    if k == 1   # node with 1 state only
        return DiscreteProbability[1.0]
    end
    # No observations means no confidence region at all, so report the vacuous interval.
    if iszero(total)
        return DiscreteProbability[Interval(0.0, 1.0) for _ in 1:k]
    end
    # all the other cases are covered by the formula
    A = quantile(Chisq(k - 1), estimator.confidence)
    return DiscreteProbability[_qh_interval(cnt, total, A) for cnt in counts]
end

# The two roots of θ^2 * (total + A) - θ * (2 * cnt + A) + cnt^2 / total = 0, the projection of the
# chi-square ellipsoid onto one coordinate. At cnt = 0 the algebra gives exactly 0 and at
# cnt = total exactly 1, so the clamp absorbs floating-point dust rather than doing any modelling;
# `max(..., 0)` guards the root against a fractional count drifting a hair past `total`.
function _qh_interval(cnt::Real, total::Real, A::Real)
    spread = sqrt(max(A * (A + 4 * cnt * (total - cnt) / total), 0.0))
    denominator = 2 * (total + A)
    return Interval(
        clamp((A + 2 * cnt - spread) / denominator, 0.0, 1.0),
        clamp((A + 2 * cnt + spread) / denominator, 0.0, 1.0)
    )
end
