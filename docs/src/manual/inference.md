# Inference

Inference answers probabilistic queries: given some observed **evidence**, what is the
distribution over one or more **query** variables? [`infer`](@ref) computes this posterior
exactly, by variable elimination, on a discrete network — a [`BayesianNetwork`](@ref) or a
[`CredalNetwork`](@ref) [jensen2007bayesian](@cite). An [`EnhancedBayesianNetwork`](@ref) is
not queried directly: it is first [`reduce`](@ref)d to one of these (see the
[Reduction & Reliability Analysis](reduction.md) chapter).

Evidence is given as an `Evidence` — a `Dict{Symbol,Symbol}` mapping node names to observed
states — and the query as a single node name or a vector of them.

```@example inference
using EnhancedBayesianNetworks # hide
W = DiscreteNode(:W); W[:W => :sunny] = 0.5; W[:W => :cloudy] = 0.5
S = DiscreteNode(:S, [:W])
S[:W => :sunny,  :S => :on] = 0.9; S[:W => :sunny,  :S => :off] = 0.1
S[:W => :cloudy, :S => :on] = 0.2; S[:W => :cloudy, :S => :off] = 0.8
bn = BayesianNetwork([W, S]); add_child!(bn, :W, :S); order!(bn)

infer(bn, :S, Evidence(:W => :sunny))       # posterior P(S | W = sunny)
```

The result is a [`Posterior`](@ref): the labelled probability table over the query, together
with the schema, query, and evidence used to produce it. Passing an empty `Evidence()` returns
the prior marginal:

```@example inference
infer(bn, :S, Evidence())                   # prior marginal P(S)
```

The query must not overlap the evidence, and both must name states that actually exist in the
network.

## Variable elimination

[`infer`](@ref) uses **variable elimination** [zhang_simple](@cite): the network's conditional
probability tables become *factors*, the evidence restricts every factor that mentions an
observed variable to its observed state, and each non-query, non-evidence variable is then
*summed out* in turn — the factors mentioning it are multiplied together and the variable is
marginalised away. Multiplying the surviving factors and normalising yields the posterior over
the query. The cost of the computation is dominated by the largest intermediate factor built
along the way, which depends heavily on the **order** in which variables are eliminated.

Formally, write the joint distribution as a product of **factors** (potentials) — initially one
per node, its CPT ``\phi_i = p(y_i \mid \mathrm{Pa}(Y_i))``:

```math
p(y_1, \dots, y_n) = \prod_{i=1}^{n} \phi_i .
```

To answer a query over the variables ``Q`` given evidence ``E = e``, every factor is first
*restricted* to the observed states, ``\phi_i\big|_{E=e}``. Each remaining variable that is
neither in ``Q`` nor in ``E`` is then **eliminated** with the two elementary factor operations —
the *product* of all factors that mention it, followed by *marginalization*, i.e. summing the
variable out of that product. Eliminating a variable ``Y_k`` yields a single new factor

```math
\psi \;=\; \sum_{y_k}\; \prod_{i\,:\, Y_k \,\in\, \operatorname{scope}(\phi_i)} \phi_i ,
```

which no longer mentions ``Y_k`` and replaces the factors it was built from. Once every
non-query, non-evidence variable has been eliminated, the product of the surviving factors is the
unnormalized joint ``p(Q, E=e)``; dividing by its total gives the posterior:

```math
p(Q \mid E = e) \;=\;
\frac{\displaystyle \sum_{\mathbf{h}} \prod_i \phi_i\big|_{E=e}}
     {\displaystyle \sum_{Q} \sum_{\mathbf{h}} \prod_i \phi_i\big|_{E=e}} ,
```

where ``\mathbf{h}`` ranges over the variables other than ``Q`` and ``E``, and the denominator is
the normalizing constant ``p(E=e)``.

## Elimination order

The elimination order is chosen greedily on the network's *interaction graph* (the moral
graph): repeatedly eliminate the remaining node with the lowest score, adding the fill-in edges
its removal induces, until none are left. Ren et al. [ren_bayesian_2022](@cite) survey heuristics
for constructing a good order and find the *minimum-increase-in-complexity* search — greedily
removing the node whose elimination least increases the problem's complexity — to outperform the
alternatives.

EnhancedBayesianNetworks.jl adapts this idea into a **mixed scored function**. Rather than
committing to a single complexity measure, it exposes two and combines them:

- [`fill_score`](@ref) — a *min-fill* heuristic: the ratio of fill-in edges an elimination would
  *add* to the edges it would *remove*, favouring nodes that introduce few new dependencies.
- [`factor_score`](@ref) — a *min-factor* heuristic: the size of the factor an elimination would
  create, the product of the state-space sizes of the node and its current neighbours.

Writing ``N(Y)`` for the current neighbours of a node ``Y`` in the interaction graph ``(V, E)``, and
``|\mathrm{dom}(Y')|`` for the number of states of a neighbour ``Y'``, the two scores are

```math
s_{\mathrm{fill}}(Y) =
\frac{\bigl|\{\, \{a,b\} \subseteq N(Y) \;:\; \{a,b\} \notin E \,\}\bigr|}{|N(Y)|},
```
```math
s_{\mathrm{factor}}(Y) = |\mathrm{dom}(Y)| \prod_{Y' \in N(Y)} |\mathrm{dom}(Y')| ,
```

with ``s_{\mathrm{fill}}(Y) = 0`` when ``Y`` has no neighbours. The numerator of ``s_{\mathrm{fill}}``
counts the neighbour pairs not yet adjacent (the fill-in edges elimination would add) and its
denominator is the degree ``|N(Y)|`` (the edges it would remove); ``s_{\mathrm{factor}}`` is the
number of entries of the factor that elimination would build.

The default, [`fill_factor_score`](@ref), is the **mix** of the two: a tuple
`(fill_score, factor_score, node)` compared lexicographically, so the min-fill score decides,
ties are broken by the smaller resulting factor, and the node id breaks any remaining tie for a
deterministic order. Either single heuristic can be selected instead by passing it as the last
argument to [`infer`](@ref):

```@example inference
infer(bn, :S, Evidence(:W => :sunny), fill_score)   # min-fill ordering only
```

The chosen heuristic changes only the *order* of elimination, never the computed posterior —
exact inference is invariant to it; the order affects only how much work is done to reach the
answer.

## Complete-scenario probability

When every node is observed there is nothing to marginalise, and the joint probability of the
full scenario is just the product of each node's conditional entry given its parents.
[`joint_probability`](@ref) computes this directly, without running variable elimination:

```@example inference
joint_probability(bn, Evidence(:W => :sunny, :S => :on))   # 0.5 * 0.9 = 0.45
```

It requires a **complete** scenario — one state per node; for partial evidence or marginals,
use [`infer`](@ref).

## Credal inference

On a [`CredalNetwork`](@ref) the local tables are interval-valued, so the network stands for a
whole credal set of Bayesian networks [cozman_credal_2000](@cite). [`infer`](@ref) then returns
a [`CredalPosterior`](@ref) carrying **lower** and **upper** posterior probabilities. These are
obtained by enumerating the *extreme* Bayesian networks of the credal set — every combination of
the interval endpoints of its imprecise nodes — running variable elimination on each, and taking
the element-wise minimum and maximum over the resulting posteriors.

```@example inference
Wc = DiscreteNode(:Wc); Wc[:Wc => :sunny] = 0.5; Wc[:Wc => :cloudy] = 0.5
Sc = DiscreteNode(:Sc, [:Wc])
Sc[:Wc => :sunny,  :Sc => :on] = Interval(0.8, 0.95); Sc[:Wc => :sunny,  :Sc => :off] = Interval(0.05, 0.2)
Sc[:Wc => :cloudy, :Sc => :on] = 0.2;                 Sc[:Wc => :cloudy, :Sc => :off] = 0.8
cn = CredalNetwork([Wc, Sc]); add_child!(cn, :Wc, :Sc); order!(cn)

infer(cn, :Sc, Evidence(:Wc => :sunny))     # CredalPosterior: [lower, upper] bounds
```

The cost grows with the number of extreme networks — one per combination of interval endpoints —
so credal inference is exact but exponential in the number of imprecise entries.

## Sampling

Besides computing posteriors, a discrete network can be **sampled** with [`sample`](@ref), which
draws realizations rather than marginals:

- `sample(bn::BayesianNetwork, n)` performs **ancestral sampling** — it draws `n` joint samples
  by visiting the nodes in topological order and drawing each from its CPT given its
  already-sampled parents. The result is a `DataFrame` with one column per node.
- `sample(node::DiscreteNode, evidence::Evidence)` draws a single state of one node, given an
  `Evidence` that fixes its parents.

```@example inference
sample(bn, 4)                               # 4 joint draws, one row per sample
```

Sampling a node whose entries are imprecise raises an error — there is no single distribution to
draw from — so `sample` applies to precise (Bayesian) networks and nodes.
