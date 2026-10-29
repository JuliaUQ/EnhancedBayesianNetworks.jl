# Discretization

[Reduction](reduction.md) eliminates continuous nodes, so a continuous node's posterior cannot be recovered afterward.
To keep evidence observable on a continuous node and to let it enter discrete inference, it can be **discretized** [straub_bayesian_2010](@cite): its support is partitioned at a list of edges into a discrete node plus a conditioned continuous remainder.
The strategy is attached to the node and depends on its position:

- a **root** carries an [`ExactDiscretization`](@ref): the discrete probabilities follow *exactly* from the node's distribution, because the marginal is known;
- a **child** carries an [`ApproximatedDiscretization`](@ref): the marginal is not generally available, so the tails are approximated (a uniform assumption over each bounded discretization-interval and an exponential assumption, of scale `λ`, over an unbounded one).

Both strategies also carry two keyword settings, `technique` and `mesh`, which decide how the probability mass of a discretization-interval is obtained when the entry is a [`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`) and are ignored otherwise; they are described under [Imprecise node](@ref) below.

Throughout, a **discretization-interval** means one cell of that partition — never an [`Interval`](@extref `UncertaintyQuantification.Interval`) entry, which is a different thing — and ``x_{ik}^-``, ``x_{ik}^+`` denote the lower and upper edges of the ``k``-th one.

## Precise node

The scheme follows [straub_bayesian_2010](@cite). The node is replaced by a **discrete surrogate**, carrying one state per discretization-interval, and a **continuous residual** ``X_i'``, carrying what is left of the variable inside the discretization-interval that was observed.

**The discrete surrogate.** The probability of state ``k`` is the mass the node's CDF puts on the ``k``-th discretization-interval,

```math
p_{ik} = F_{X_i}(x_{ik}^+) - F_{X_i}(x_{ik}^-).
```

On a **root** this is exact, ``F_{X_i}`` being the node's own marginal. On a **child** the same difference is taken once per parent-state scenario, from that scenario's distribution, so the surrogate inherits the node's parents.

**The continuous residual.** Which form the residual takes depends on the node's **position**, not on ``F_{X_i}``.

A **root** conditions its own CDF on the discretization-interval, exactly:

```math
F_{X_i'}(x_i \mid k) =
\begin{cases}
0 & x_i \le x_{ik}^-, \\
\dfrac{F_{X_i}(x_i) - F_{X_i}(x_{ik}^-)}{F_{X_i}(x_{ik}^+) - F_{X_i}(x_{ik}^-)} & x_{ik}^- \le x_i \le x_{ik}^+, \\
1 & x_i \ge x_{ik}^+.
\end{cases}
```

A **child** cannot use the (unknown) marginal, so each bounded discretization-interval is filled with a **uniform** assumption,

```math
F_{X_i'}(x_i \mid k) =
\begin{cases}
0 & x_i \le x_{ik}^-, \\
\dfrac{x_i - x_{ik}^-}{x_{ik}^+ - x_{ik}^-} & x_{ik}^- \le x_i \le x_{ik}^+, \\
1 & x_i \ge x_{ik}^+,
\end{cases}
```

and each unbounded tail with an **exponential** assumption of scale ``\lambda`` (the `λ` argument), pointing away from the discretization-interval's one finite edge. Above a lower edge ``x_{ik}^-``,

```math
F_{X_i'}(x_i \mid k) =
\begin{cases}
0 & x_i \le x_{ik}^-, \\
1 - \exp\!\left[-(x_i - x_{ik}^-) / \lambda\right] & x_i > x_{ik}^-,
\end{cases}
```

and, mirrored, below an upper edge ``x_{ik}^+``,

```math
F_{X_i'}(x_i \mid k) =
\begin{cases}
\exp\!\left[-(x_{ik}^+ - x_i) / \lambda\right] & x_i \le x_{ik}^+, \\
1 & x_i > x_{ik}^+.
\end{cases}
```

A larger ``\lambda`` is a longer tail: it is the exponential's **scale**, so the residual sits a mean distance ``\lambda`` beyond the finite edge.

```@example discretization
using EnhancedBayesianNetworks # hide
Tr = ContinuousNode(:Tr, Normal(), ExactDiscretization([-2.0, 0.0, 2.0]))   # root
```

```@example discretization
# child: discretization-interval edges plus the exponential tail scale (here 1.5)
Cd = ContinuousNode(:Cd, [:W], ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5))
```

## Imprecise node

When the node is **imprecise** the same partition is used, and the split into surrogate and residual is the same; only the surrogate's probabilities change shape, becoming intervals rather than numbers.

**The discrete surrogate.** The mass wanted is the same, ``p_{ik} = F_{X_i}(x_{ik}^+) - F_{X_i}(x_{ik}^-)``, but a **probability box** is not one CDF: it is a family ``\{F_\theta : \theta \in \Theta\}``, each member precise, indexed by a parameter box ``\Theta``. Every member assigns its own mass to the interval, so the mass is itself an interval: the range of that mass over the family,

```math
p_{ik} = \Big[\; \min_{\theta \in \Theta}\big(F_{\theta}(x_{ik}^+) - F_{\theta}(x_{ik}^-)\big),\;\;
                \max_{\theta \in \Theta}\big(F_{\theta}(x_{ik}^+) - F_{\theta}(x_{ik}^-)\big) \;\Big].
```

This is the default, selected by `technique = :optimised`, and the two bounds are found by direct search over ``\Theta`` (`MeshAdaptiveDirectSearch`), one member of the family at a time, down to a terminal mesh size of `mesh`. A smaller `mesh` is tighter and slower.

Evaluating one member at a time is what makes the result correct, and the two shortcuts one might reach for instead are both wrong. The p-box's own envelopes ``\underline{F}`` and ``\overline{F}`` are built *pointwise* over ``\Theta``, so ``\underline{F}(x_{ik}^+)`` and ``\underline{F}(x_{ik}^-)`` may be attained by different members; differencing them mixes two distributions and bounds the mass of neither. And the mass need not be monotone in ``\theta``, so evaluating only the corners of ``\Theta`` misses an interior optimum — for a Normal with an interval mean, the mass of a discretization-interval is largest when the mean sits at its midpoint, which is an interior point of ``\Theta``.

When the family is not described by a parameter box, or the search is too slow, `technique = :distribution_free` differences *opposite* envelopes instead,

```math
p_{ik} = \Big[\; \max\big(0,\; \underline{F}(x_{ik}^+) - \overline{F}(x_{ik}^-)\big),\;\;
                \min\big(1,\; \overline{F}(x_{ik}^+) - \underline{F}(x_{ik}^-)\big) \;\Big],
```

which is valid for *any* CDF lying between the envelopes. On a parametric p-box it therefore **encloses** the range above: never tighter, usually wider, and equal only on a discretization-interval where the two opposite envelopes happen to be attained by the same member — on a tail, say, where the mass is monotone in ``\theta``. The clamps are not cosmetic: the lower expression goes negative as soon as the envelopes are far apart, and an [`Interval`](@extref `UncertaintyQuantification.Interval`) only rejects a lower bound above its upper one, so a negative probability would otherwise pass. `mesh` is unused here.

Both settings live on the discretization strategy, so they are chosen per node, and both are consulted only for a probability-box entry — a precise distribution and a bare interval ignore them.

!!! note "Parametric probability boxes only, for now"
    The only probability box that can be written as a node's entry is a **parametric** one: a [`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`) is a distribution family plus a box of parameter values, so every member of it belongs to that family. A **distribution-free** box — any CDF running between two envelopes, with no family behind it — has no dedicated type yet; it is planned for [UncertaintyQuantification.jl](https://github.com/JuliaUQ/UncertaintyQuantification.jl). Two things follow from that.

    First, a continuous functional node fed by an imprecise parent produces exactly such a distribution-free object, which is why it is kept as a lower/upper pair of empirical CDFs and cannot itself be discretized — see [Continuous Functional nodes](@ref).

    Second, `technique = :distribution_free` is the method those boxes will need once they exist, because it assumes nothing beyond the envelopes. Until then every p-box reaching it is parametric, and on a parametric p-box it is always an enclosure rather than the answer: correct, cheaper, and no tighter than `:optimised` on any discretization-interval.

For a bare **interval** entry no CDF is available at all — only the support — so every discretization-interval receives the vacuous mass ``p_{ik} = [0, 1]``. Either way the surrogate node becomes imprecise (credal), and that is what carries the imprecision into inference.

!!! note "Vacuous masses and conditioning"
    A node discretized from a bare **interval** is *vacuous*: masses of ``[0, 1]`` on every discretization-interval constrain nothing beyond summing to one, so the node's credal set is the whole probability simplex over its states and its extreme points are the degenerate distributions putting all the mass on a single discretization-interval. Evidence on such a node then has probability zero under some of those extreme points, and meaningful bounds are only obtained under regular extension — see [Credal inference](@ref). A [`ProbabilityBox`](@extref `UncertaintyQuantification.ProbabilityBox`) can narrow these mass intervals, but it avoids zero-probability extremes only when the lower mass of the observed discretization-interval is strictly positive.

**The continuous residual.** The formulas are exactly the ones given for a precise node, and as there it is the node's **position** that selects them — the entry's imprecision plays no part.

A **root** conditions its entry on the discretization-interval, which preserves whatever kind of entry it was: a distribution is truncated, a probability box is rebuilt on the narrower support and stays a box, and a bare `Interval` becomes the sub-interval ``[x_{ik}^-, x_{ik}^+]``. A **child** uses the uniform or exponential-tail approximation unchanged, so its residual is a precise distribution however imprecise the entry was.

The asymmetry is deliberate: on a root the imprecision lives in both the surrogate probabilities and the residual; on a child it is pushed entirely into the surrogate's interval probabilities, leaving a precise residual. That is why an imprecise child feeding a functional node needs a single-loop simulation — the imprecision has moved into the discrete surrogate and no longer reaches the functional node as an input.

```@example discretization
# imprecise root: an interval distribution, discretized exactly
Ti = ContinuousNode(:Ti, Interval(-2.0, 2.0), ExactDiscretization([-2.0, 0.0, 2.0]))
```

```@example discretization
# imprecise child: an interval entry plus the exponential tail scale
Ci = ContinuousNode(:Ci, [:W], ApproximatedDiscretization([-1.0, 0.0, 1.0], 1.5))
Ci[:W => :sunny] = Interval(-1.0, 1.0)
Ci
```

A probability-box root, whose discretization-interval masses are optimised over the parameter box at the default mesh:

```@example discretization
pbox = ProbabilityBox{Normal}(Dict(:μ => Interval(0.0, 1.0), :σ => 1))
Tp = ContinuousNode(:Tp, pbox, ExactDiscretization([-1.0, 0.0, 1.0]))
```

The same p-box taking the enclosure instead. A node reports these settings only when they depart from the defaults, so nothing extra appears above and a line does here; a non-default `mesh` shows up the same way:

```@example discretization
Tf = ContinuousNode(:Tf, pbox, ExactDiscretization([-1.0, 0.0, 1.0]; technique = :distribution_free))
```
