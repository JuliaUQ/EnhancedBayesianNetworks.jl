# The frequencies a single parent configuration contributes to one node's CPT column. Under complete data only `counts` and `total` are non-zero, and the estimators reduce to their familiar form. The other three carry what an incomplete row still tells you, and are what the robust Bayesian estimator (Ramoni & Sebastiani 2001) needs to bound an estimate over every consistent completion of the data.
# Parametric rather than fixed to `Int` because not every frequency is a count of rows: an EM step divides a row between the states it might have taken, so its M-step reaches `_estimate` with sums of responsibilities. `T` is `Int` for data counted off a data frame and `Float64` when EM is employed, and the five fields share it, so a caller mixing the two must convert first.
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

Abstract supertype of the estimators accepted by [`learn`](@ref) and by the `estimator` keyword of [`add_node!`](@ref). An estimator fixes *how* the probabilities of one parent configuration are derived from the data, and with it the kind of network returned: a precise estimator gives one probability per node state and a [`BayesianNetwork`](@ref), an imprecise one an `Interval` per node state and a [`CredalNetwork`](@ref).

Every estimator works on **one parent configuration**: one node together with one configuration of its parents' states. Three quantities describe it:

  - `k`, how many states the node has.
  - `cnt`, how many data rows match the parent configuration *and* have the node in the state being estimated.
  - `total`, how many such rows there are over all `k` states, so `total` is always `sum(cnt)`. On incomplete data it counts only the rows where the node *and* every one of its parents were observed.

Available estimators are [`PreciseDirichletModel`](@ref), [`ImpreciseDirichletModel`](@ref) and [`QuesenberryHurst`](@ref).
"""
abstract type ParameterEstimator end

# The two families of estimators, internal: users only ever construct the concrete ones. `_verify_missingness` reads them to choose between expectation maximisation, for precise networks, and bounding over the completions of the data, for imprecise ones.
abstract type PreciseEstimator <: ParameterEstimator end
abstract type ImpreciseEstimator <: ParameterEstimator end

"""
    PreciseDirichletModel(α = 0)

CPT entries as `(cnt + α) / (total + α * k)`, the mean of a symmetric Dirichlet posterior with pseudo-count `α ≥ 0` in each of the `k` states. Here `k` is the number of states of the node, `cnt` the number of rows matching the parent configuration with the node in the state being estimated, and `total` the number of rows matching it in any state.

`α = 0` is maximum likelihood, the relative frequency, and gives exactly `0` to a state never observed in a configuration; any `α > 0` is additive smoothing, Laplace smoothing at `α = 1`, pulling towards the uniform. Use [`ImpreciseDirichletModel`](@ref) to report that uncertainty as an interval rather than smooth it away. A parent configuration absent from the data gives the uniform distribution.

On incomplete data [`learn`](@ref) reaches this estimator through expectation maximisation, which assumes the data are missing at random; `max_iter` and `tol` control the iteration.

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

CPT entries as the interval `[cnt / (total + s), (cnt + s) / (total + s)]`, with `s > 0` the prior strength: `s` extra observations whose states are left unspecified, placed outside the estimated state for the lower bound and inside it for the upper. Here `k` is the number of states of the node, `cnt` the number of rows matching the parent configuration with the node in the state being estimated, and `total` the number of rows matching it in any state. `s = 1` and `s = 2` are the usual choices; larger `s` gives wider intervals.

It is [`PreciseDirichletModel`](@ref) with the prior mean left free instead of pinned to the uniform, so the interval always contains `PreciseDirichletModel(s / k)`. Two properties make it the default for credal learning: the credal set is never empty, the lower bounds of a configuration summing to at most `1` and the upper bounds to at least `1`; and a configuration absent from the data gives `[0, 1]` with no special case, which [`infer`](@ref) handles through the regular extension.

On incomplete data the interval widens to the envelope of the estimates over every completion of the data consistent with what was observed, so nothing is assumed about *why* a value is absent. In particular not that it is missing at random. Each bound gains a virtual frequency counting the incomplete rows it may ascribe to the configuration; a row contradicted by any observed parent is excluded however much else is unknown.

References: Walley, P. (1996), *Inferences from Multinomial Data: Learning About a Bag of Marbles*, Journal of the Royal Statistical Society Series B 58(1), 3–34. For incomplete data, Ramoni, M. & Sebastiani, P. (2001), *Robust Learning with Missing Data*, Machine Learning 45(2), 147–170, whose robust Bayesian estimator gives the bounds in closed form.

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

CPT entries as the simultaneous confidence region for multinomial proportions projected onto one coordinate, obtained by inverting Pearson's score test:

    θ = [(A + 2 * cnt) ± sqrt(A * (A + 4 * cnt * (total - cnt) / total))] / (2 * (total + A))

with `A = quantile(Chisq(k - 1), confidence)`. Here `k` is the number of states of the node, `cnt` the number of rows matching the parent configuration with the node in the state being estimated, and `total` the number of rows matching it in any state. The `k - 1` degrees of freedom are what make the coverage **simultaneous** over the node's states rather than one state at a time; at `k = 2` this is exactly the Wilson score interval. A configuration absent from the data gives `[0, 1]`, and coherence holds whenever `total > 0`.

Unlike [`ImpreciseDirichletModel`](@ref) this is a frequentist confidence region read as a credal set: it carries repeated-sampling coverage, not a behavioural interpretation, so results should say which of the two produced them.

[`learn`](@ref) refuses this estimator on incomplete data, where the envelope over the completions has no closed form and the simultaneity above would no longer hold; use [`ImpreciseDirichletModel`](@ref).

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
# `k = length(frequencies.counts)` is read off the record, so anything depending only on `k`, notably the chi-square quantile, is computed once per parents' configuration rather than once per node state.
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
