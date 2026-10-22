# The frequencies a single parent configuration contributes to one node's CPT column. Under complete data only `counts` and `total` are non-zero, and the estimators reduce to their familiar form. The other three carry what an incomplete row still tells you, and are what the robust Bayesian estimator (Ramoni & Sebastiani 2001) needs to bound an estimate over every consistent completion of the data.
# Parametric rather than fixed to `Int` because not every frequency is a count of rows: an EM step divides a row between the states it might have taken, so its M-step reaches `_estimate` with sums of responsibilities. `T` is `Int` for data counted off a data frame and `Float64` there, and the five fields share it, so a caller mixing the two must convert first.
struct ConfigurationCounts{T <: Real}
    counts::Vector{T}           # n(x_k | pi_j): parents and node both observed
    total::T                    # n(pi_j) = sum(counts), complete cases only
    node_missing::T             # n(? | pi_j): parents observed, node missing
    parents_missing::Vector{T}  # n(x_k | ?): node observed, parents consistent but incomplete
    both_missing::T             # n(? | ?): neither observed, parents consistent
end

# ==================================================================================================
# Parameter estimators
#
# An estimator answers one question: given the data counts of a single parent configuration, what goes in that row of the CPT? Precise estimators return numbers and yield a `BayesianNetwork`; imprecise ones return `Interval`s and yield a `CredalNetwork`.
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

On an **incomplete** data set the same interval widens to the envelope of the estimates over every completion of the data consistent with what was observed, so nothing is assumed about *why* a value is absent — in particular not that it is missing at random, the assumption the imprecise framework exists to avoid. Each bound gains a *virtual frequency*, the number of incomplete rows it may ascribe to the parent configuration. A row whose node value is unknown could have been any state, so it widens both ends; a row whose node value is known but whose parents are not raises the upper bound of its own state and the lower bound of every other one, since ascribing it here inflates their denominator without touching their numerator. A row contradicted by any *observed* parent is excluded however much else is unknown.

Two consequences are worth knowing. With no missing values both virtual frequencies vanish and the expression above is recovered exactly, and every additional missing value can only widen the interval, so the estimate is **monotone in the information available** — losing data can never make it more confident. Against that, the two bounds of a given entry are attained by *different* completions, so for `k > 2` the resulting CPT is an outer approximation of the credal set the data really licenses: correct, in that it contains the truth, but wider than strictly necessary. For `k = 2` it is exact.

References: Walley, P. (1996), *Inferences from Multinomial Data: Learning About a Bag of Marbles*, Journal of the Royal Statistical Society Series B 58(1), 3–34. For the treatment of incomplete data, Ramoni, M. & Sebastiani, P. (2001), *Robust Learning with Missing Data*, Machine Learning 45(2), 147–170, whose robust Bayesian estimator gives the bounds above in closed form.

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
# _estimate(estimator, frequencies) -> one parents' configuration, whole discrete node conditional probabilities given parents' configuration
#
# `frequencies` is the `ConfigurationCounts` of a single parent configuration: `frequencies.counts[i]` is the number of rows matching that configuration whose node state is the i-th of the node's domain, and `frequencies.total == sum(frequencies.counts)` is the normaliser. The whole record is passed rather than those two fields alone because a conservative treatment of missing data also needs the three incomplete-case frequencies, which are zero throughout on complete data and leave every formula below untouched.
#
# `k = length(frequencies.counts)` is read off the record, so anything depending only on `k` — notably the chi-square quantile — is computed once per parents' configuration rather than once per node state.
#
# Each method owns its own degenerate cases; the fitting loop has none.
# --------------------------------------------------------------------------------------------------

function _estimate(estimator::PreciseDirichletModel, frequencies::ConfigurationCounts)
    k = length(frequencies.counts)
    if k == 1   # node with 1 state only
        return DiscreteProbability[1.0]
    end
    α = estimator.α
    # MLE and parents configuration not present
    if (iszero(frequencies.total) && iszero(α))
        return DiscreteProbability[1 / k for _ in 1:k]
    end
    # all the other cases are covered; the clamp is defensive, since with exact counts the ratio already lies in [0, 1], and it earns its keep only where `cnt` is not an exact count, as in the EM steps
    return DiscreteProbability[clamp((cnt + α) / (frequencies.total + α * k), 0.0, 1.0) for cnt in frequencies.counts]
end

function _estimate(estimator::ImpreciseDirichletModel, frequencies::ConfigurationCounts)
    k = length(frequencies.counts)
    if k == 1   # node with 1 state only
        return DiscreteProbability[1.0]
    end
    s = estimator.s
    # Virtual frequencies (Ramoni & Sebastiani 2001, Eqs. 3-4): how many incomplete rows each bound may ascribe to this configuration. Rows whose node value is unknown are shared, since they could have been any state; a row whose node value is known but whose parents are not counts towards the upper bound of its own state and towards the lower bound of every other one, because ascribing it here inflates their denominator without touching their numerator.
    shared = frequencies.node_missing + frequencies.both_missing
    ambiguous = sum(frequencies.parents_missing)
    upper_virtual = shared .+ frequencies.parents_missing
    lower_virtual = shared .+ ambiguous .- frequencies.parents_missing
    # all the other cases are covered; the clamp is defensive, as above
    return DiscreteProbability[
        Interval(
            clamp(frequencies.counts[i] / (frequencies.total + lower_virtual[i] + s), 0.0, 1.0),
            clamp((frequencies.counts[i] + upper_virtual[i] + s) / (frequencies.total + upper_virtual[i] + s), 0.0, 1.0)
        ) for i in 1:k
    ]
end

function _estimate(estimator::QuesenberryHurst, frequencies::ConfigurationCounts)
    k = length(frequencies.counts)
    if k == 1   # a node with a single state is certain whatever the estimator, and there is nothing left to bound
        return DiscreteProbability[1.0]
    end
    # No observations means no confidence region at all, so report the vacuous interval.
    if iszero(frequencies.total)
        return DiscreteProbability[Interval(0.0, 1.0) for _ in 1:k]
    end
    # all the other cases are covered by the formula
    A = quantile(Chisq(k - 1), estimator.confidence)
    return DiscreteProbability[_qh_interval(cnt, frequencies.total, A) for cnt in frequencies.counts]
end

# At cnt = 0 the algebra gives exactly 0 and at cnt = total exactly 1, so the clamp absorbs floating-point dust rather than doing any modelling;
function _qh_interval(cnt::Real, total::Real, A::Real)
    spread = sqrt(max(A * (A + 4 * cnt * (total - cnt) / total), 0.0))
    denominator = 2 * (total + A)
    return Interval(
        clamp((A + 2 * cnt - spread) / denominator, 0.0, 1.0),
        cnt == total ? 1.0 : clamp((A + 2 * cnt + spread) / denominator, 0.0, 1.0)
    )
end
