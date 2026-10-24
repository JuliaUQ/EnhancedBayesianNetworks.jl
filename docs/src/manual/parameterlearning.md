# Parameter Learning

Parameter learning fills in the conditional probability tables (CPTs) of a network from **data**, given its structure.
You declare which nodes exist and how they depend on one another, and [`learn`](@ref) estimates every CPT from a `DataFrame` of observations.
The **estimator** decides what kind of network comes out: a precise estimator gives a [Bayesian Network](@ref) (BN), an imprecise one gives a [Credal Network](@ref) (CN) whose entries are intervals.

## Declaring the structure

The structure is a [`DirectAcyclicGraph`](@ref): nodes and edges, without any probability.
Nodes are declared with [`add_node!`](@ref), parents first, and the edges are wired as each node is added, so a cycle cannot be built.

```@example learning
using EnhancedBayesianNetworks # hide
using DataFrames # hide
dag = DirectAcyclicGraph()
add_node!(dag, :W; states = [:foggy])   # root, :foggy kept even though it never occurs in the data
add_node!(dag, :R; parents = [:W])      # edge :W -> :R
add_node!(dag, :G; parents = [:R])      # edge :R -> :G
nothing # hide
```

A node's **domain** is the set of states observed in the data, together with any declared through the `states` keyword.
Declaring a state is the way to keep one that the data happen not to contain: under maximum likelihood it gets probability exactly zero, while the other estimators assign it whatever they assign an unobserved state.

The data are a `DataFrame` with one `Symbol` column per node; columns that are not nodes of the DAG are ignored with a warning.

```@example learning
df = DataFrame(
    W = [:sunny, :sunny, :sunny, :sunny, :sunny, :sunny, :cloudy, :cloudy, :cloudy, :cloudy],
    R = [:no, :no, :no, :no, :no, :yes, :yes, :yes, :yes, :no],
    G = [:dry, :dry, :dry, :wet, :dry, :wet, :wet, :wet, :wet, :dry],
)
```

## Learning the parameters

Learning with a known structure splits into **independent local problems**.
For complete data the likelihood of a network factorises over its nodes and, within each node ``X``, over the configurations ``\pi_j`` of its parents:

```math
L(\theta \mid \mathcal{D}) = \prod_{X} \prod_{j} \prod_{i=1}^{k} \theta_{ij}^{\,n_{ij}},
```

where ``\theta_{ij} = P(X = x_i \mid \mathrm{Pa}(X) = \pi_j)``, ``k`` is the number of states of ``X``, ``n_{ij}`` counts the rows where the parents are in configuration ``\pi_j`` and ``X`` is in state ``x_i``, and ``n_j = \sum_i n_{ij}`` counts the rows with configuration ``\pi_j`` at all.
Each column ``(\theta_{1j}, \dots, \theta_{kj})`` of a CPT can therefore be estimated on its own, from its counts ``n_{1j}, \dots, n_{kj}`` alone.
An estimator is precisely a rule for that one column, and the same rule is applied to every column of every node.

[`learn`](@ref) takes the estimator as its third argument:

```@example learning
bn = learn(dag, df, PreciseDirichletModel())
```

The returned network is fully specified but not ordered; call [`order!`](@ref) before inference or sampling.

### Precise estimates

[`PreciseDirichletModel`](@ref)`(α)` estimates each entry as

```math
\hat{\theta}_{ij} = \frac{n_{ij} + \alpha}{n_j + k\,\alpha}, \qquad \alpha \ge 0 .
```

This is the posterior mean of ``\theta_{\cdot j}`` under a symmetric Dirichlet prior with ``\alpha`` pseudo-counts per state, the conjugate prior of the multinomial, so the posterior is again a Dirichlet and its mean has this closed form.

- ``\alpha = 0`` is **maximum likelihood**, the relative frequency ``n_{ij} / n_j``. A state never observed under a configuration gets probability exactly zero.
- ``\alpha = 1`` is **Laplace smoothing**, and any other ``\alpha > 0`` is additive smoothing: every estimate is pulled towards the uniform ``1/k``, more strongly the fewer observations the configuration has.

A parent configuration absent from the data, ``n_j = 0``, gets the uniform distribution in both cases.
Under maximum likelihood that is a convention, since the data leave ``0/0``; with ``\alpha > 0`` it is what the formula gives.

```@example learning
bn = learn(dag, df, PreciseDirichletModel(1))
order!(bn)
bn.nodes[bn.topology[:R]]
```

The row ``W = \text{foggy}`` has no data, so it holds the uniform ``1/2``.
That number is not a measurement but the prior speaking alone, and nothing in a precise CPT tells the two apart.

### Imprecise estimates

The **imprecise Dirichlet model** [walley_inferences_1996](@cite) replaces the single prior by the set of all Dirichlet priors of strength ``s``, with any mean.
The posterior means then range over an interval, and [`ImpreciseDirichletModel`](@ref)`(s)` reports it:

```math
\theta_{ij} \in \left[\, \frac{n_{ij}}{n_j + s},\; \frac{n_{ij} + s}{n_j + s} \,\right], \qquad s > 0 .
```

The lower bound places the ``s`` prior observations outside the state ``x_i``, the upper bound places them all inside it.
The interval always contains the precise estimate with ``\alpha = s / k``, and its width ``s / (n_j + s)`` shrinks as data accumulate, so the imprecision measures how little the data say.
Walley recommends ``s = 1`` or ``s = 2``.

Two properties make it the natural default for credal learning:

- the credal set of every column is never empty: the lower bounds sum to at most ``1`` and the upper bounds to at least ``1``;
- a configuration absent from the data gets the **vacuous** interval ``[0, 1]``, which states that nothing is known, rather than a uniform distribution that looks like knowledge.

```@example learning
cn = learn(dag, df, ImpreciseDirichletModel())
order!(cn)
cn.nodes[cn.topology[:R]]
```

[`QuesenberryHurst`](@ref)`(confidence)` gives intervals of a different nature: the **simultaneous confidence intervals** for multinomial proportions of [quesenberry_large_1964](@cite), obtained by inverting Pearson's ``\chi^2`` test.
With ``A`` the ``\chi^2_{k-1}`` quantile at the chosen confidence level,

```math
\theta_{ij} \in \frac{A + 2 n_{ij} \pm \sqrt{A \left( A + 4\, n_{ij} (n_j - n_{ij}) / n_j \right)}}{2\,(n_j + A)} .
```

The ``k - 1`` degrees of freedom make the coverage hold for all ``k`` states of the column **at once**, not one state at a time; for a binary node the interval reduces to the Wilson score interval.
A configuration absent from the data again gets ``[0, 1]``.

!!! note "Two readings of an interval"
    The two imprecise estimators answer different questions. An [`ImpreciseDirichletModel`](@ref) interval is a set of posterior beliefs: it is the range of estimates over a set of priors, and carries a behavioural interpretation. A [`QuesenberryHurst`](@ref) interval is a frequentist confidence region read as a credal set: it carries repeated-sampling coverage and no belief at all. Both give a valid [`CredalNetwork`](@ref), but results should say which one produced them.

### One estimator per node

Instead of one estimator for the whole network, each node can declare its own through the `estimator` keyword of [`add_node!`](@ref), and [`learn`](@ref) is then called without one.
This makes it possible to be imprecise only where the data are thin.
A single imprecise node makes the result a [`CredalNetwork`](@ref).

```@example learning
mixed = DirectAcyclicGraph()
add_node!(mixed, :W; states = [:foggy], estimator = ImpreciseDirichletModel())
add_node!(mixed, :R; parents = [:W], estimator = PreciseDirichletModel(1))
add_node!(mixed, :G; parents = [:R], estimator = PreciseDirichletModel(1))
learn(mixed, df)
```

It is all or nothing: a DAG where some nodes declare an estimator and others do not is rejected, and so is passing an estimator to [`learn`](@ref) for a DAG that already declares them.

## Incomplete data

Entries of the data frame may be `missing`.
What that means depends on the estimators, and [`learn`](@ref) picks the treatment from them rather than from a separate option.

### Precise networks: expectation maximisation

When every node is precise, the CPTs are fitted by **expectation maximisation** (EM) [dempster_maximum_1977](@cite).
Starting from uniform CPTs ``\theta^{(0)}``, each iteration

- **E-step**: replaces every incomplete row by all its completions, each weighted by its probability given the observed values under the current network, ``P(\text{missing} \mid \text{observed}; \theta^{(t)})``, so that the counts ``n_{ij}`` become *expected* counts;
- **M-step**: applies the node's estimator to those expected counts, giving ``\theta^{(t+1)}``. With ``\alpha = 0`` this is the maximum-likelihood step; with ``\alpha > 0`` it is the same smoothed estimate, computed from the expected counts.

The iteration stops when no CPT entry changes by more than `tol` (default ``10^{-4}``), or after `max_iter` iterations (default ``100``), in which case [`learn`](@ref) warns that it did not converge.

EM assumes the data are **missing at random**: whether a value is missing may depend on what was observed, but not on the missing value itself.
When that fails, for instance when a sensor drops out precisely when the reading is extreme, the estimates are biased, and nothing in the result shows it.
EM also converges to a local maximum of the likelihood, which need not be the global one.

```@example learning
incomplete = allowmissing(df)
incomplete[2, :R] = missing
incomplete[7, :W] = missing
incomplete[9, :G] = missing
bn_em = learn(dag, incomplete, PreciseDirichletModel(1))
order!(bn_em)
bn_em.nodes[bn_em.topology[:R]]
```

Row 2, whose rain is unknown, is split between `:no` and `:yes` in proportion to the current estimates.
Row 7, whose weather is unknown, is spread over every weather state, `:foggy` included, which is why the `:foggy` column is no longer exactly uniform: EM treats a missing value as one more thing to estimate, never as a reason for doubt.

### Imprecise networks: bounds over every completion

When every node is imprecise, [`learn`](@ref) assumes **nothing** about why the values are missing.
Each entry is bounded over every completion of the data consistent with what was observed, using the robust Bayesian estimator of [ramoni_robust_2001](@cite), whose bounds are available in closed form.

For a column ``\pi_j`` of node ``X``, the incomplete rows that could belong to it come in three kinds: rows with the parents in ``\pi_j`` and ``X`` missing (``n^{?}_{j}``), rows with ``X`` missing and the parents only partly observed but consistent with ``\pi_j`` (``n^{??}_{j}``), and rows with ``X = x_i`` observed and the parents consistent with ``\pi_j`` but incomplete (``m_{ij}``).
A row contradicted by any observed parent is excluded, however much else is unknown.
The extreme completions ascribe these rows to the configuration and the state so as to push each bound as far as it goes, which adds a **virtual count** to the [`ImpreciseDirichletModel`](@ref) formula:

```math
\theta_{ij} \in \left[\, \frac{n_{ij}}{n_j + \underline{v}_{ij} + s},\; \frac{n_{ij} + \overline{v}_{ij} + s}{n_j + \overline{v}_{ij} + s} \,\right],
\qquad
\begin{aligned}
\overline{v}_{ij} &= n^{?}_{j} + n^{??}_{j} + m_{ij}, \\
\underline{v}_{ij} &= n^{?}_{j} + n^{??}_{j} + \textstyle\sum_{l \ne i} m_{lj} .
\end{aligned}
```

For the upper bound, every row whose value of ``X`` is unknown is taken to be ``x_i``, together with every row observed as ``x_i`` that might belong to the configuration; for the lower bound, the same unknown rows are taken to be some other state, and so are the rows observed in other states.
On complete data all virtual counts are zero and the formula is the ordinary one.

```@example learning
cn_missing = learn(dag, incomplete, ImpreciseDirichletModel())
order!(cn_missing)
cn_missing.nodes[cn_missing.topology[:R]]
```

The intervals are wider than on complete data by exactly as much as the missing values could matter.
On complete data the column ``W = \text{sunny}`` gave ``[0.71, 0.86]`` for `:no`; here it gives ``[0.5, 0.86]``.
The upper bound counts row 2, rain unknown, as `:no`; the lower bound counts it as `:yes`, and also ascribes row 7, observed as `:yes` with unknown weather, to the sunny configuration.
This is the honest alternative to EM when the missing-at-random assumption cannot be defended.

### What is refused

Two combinations have no coherent answer, and [`learn`](@ref) raises an error rather than produce one:

- [`QuesenberryHurst`](@ref) on incomplete data. Its region has no closed-form envelope over the completions, and bounding each entry separately would no longer carry the stated simultaneous confidence. Use [`ImpreciseDirichletModel`](@ref) instead.
- A DAG mixing precise and imprecise nodes on incomplete data. EM is a property of the whole network, since its E-step needs the joint distribution, while bounding over the completions is local to each node, so the two cannot be combined. Declare one kind of estimator throughout, or complete the data first.

```julia
learn(dag, incomplete, QuesenberryHurst())   # error: QuesenberryHurst cannot be fitted from incomplete data
```

## Using the learned network

A learned network is an ordinary [`BayesianNetwork`](@ref) or [`CredalNetwork`](@ref), and once ordered it supports everything described in the [Inference](inference.md) chapter.

```@example learning
infer(cn, :G, Evidence(:W => :sunny))
```

On a credal network, [`infer`](@ref) runs one variable elimination per **extreme network** of the credal set.
Their number is the product, over every node and every parent configuration, of the number of vertices of that column's credal set:

```math
N = \prod_{X} \prod_{j} \left| \operatorname{ext} K_j(X) \right| ,
```

so it grows exponentially with the number of imprecise columns.
A learned network makes this concrete: every column becomes imprecise, and a node with three states and twelve parent configurations already contributes up to ``6^{12} \approx 2 \cdot 10^9`` extreme nodes on its own.
[`infer`](@ref) counts ``N`` before building any network and raises an error when it exceeds the `max_networks` keyword (default ``10^6``), instead of starting a computation that cannot finish.

!!! tip "Keeping credal inference feasible"
    Declare an imprecise estimator only on the nodes whose data are too thin to trust, with per-node estimators, and keep the rest precise. Imprecise roots and nodes with few parents and states are cheap; an imprecise node with many parent configurations is what makes ``N`` explode.
