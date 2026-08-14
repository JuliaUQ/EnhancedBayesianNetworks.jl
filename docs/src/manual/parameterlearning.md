# Parameter Learning

Parameter learning fills in the conditional probability tables of a [Bayesian Network](@ref) (BN) from **data**, given its structure. 
You supply the graph, meaning which nodes exist and how they depend on one another, and a dataset of observations.
The learned network is a fully-specified BN, ready for [inference](inference.md) once ordered.

The structure is described by a [`DirectAcyclicGraph`](@ref): nodes and edges, plus optional declared states, but no probabilities. 
It is a lightweight input type, not a network, it supports [`add_node!`](@ref), [`parents`](@ref), [`children`](@ref), and [`gplot`](@ref), but none of the
operations that need CPTs ([`order!`](@ref), [`infer`](@ref), `sample`).

## Declaring the structure

Build the DAG top-down with [`add_node!`](@ref), listing each node's `parents` (which must already be present). 
Edges are wired for you as you declare the parents.

```@example parameters_learning
using EnhancedBayesianNetworks

dag = DirectAcyclicGraph()
add_node!(dag, :W)                          # root
add_node!(dag, :R; parents = [:W])          # edge :W -> :R wired here
add_node!(dag, :P; parents = [:W])
add_node!(dag, :G; parents = [:R, :P])
```

Each node's **domain** is taken from the data at learn time. 
Any states passed to [`add_node!`](@ref) are *added* to that domain. Those are the states you want to keep even if they never occur in the dataset. 
They end up in the learned CPT with probability `0` (or an `alpha`-smoothed mass):

```@example parameters_learning
add_node!(dag, :W, [:foggy])                # :sunny/:cloudy come from data; :foggy guaranteed
```

## Learning the parameters

[`learn`](@ref) estimates the CPTs and chooses the algorithm from the data: **complete** data uses maximum likelihood, data with any `missing` entries uses Expectation-Maximization. 
It returns a BN; call [`order!`](@ref) on it before inference or sampling.

```@example parameters_learning
using DataFrames
df = DataFrame(W = [:sunny, :sunny, :cloudy, :cloudy],
               S = [:on,    :off,   :on,     :off])

dag = DirectAcyclicGraph()
add_node!(dag, :W)
add_node!(dag, :S; parents = [:W])

learned = learn(dag, df)                    # complete data -> MLE
order!(learned)
```

`alpha` is a Laplace/Dirichlet pseudo-count applied by both algorithms; `max_iter` and `tol` control EM's convergence:

```@example parameters_learning
learn(dag, df; alpha = 1)                            # add-one smoothing
learn(dag, df_with_missing; tol = 1e-6, max_iter = 500)
```

To force a specific algorithm, call it directly.

### Maximum likelihood

[`learn_parameters_mle`](@ref) is the closed-form estimator for **complete** data. 
For every node ``Y_i``, each of its states ``s``, and each configuration ``c`` of its parents ``\mathrm{Pa}(Y_i)``, it sets

```math
P(Y_i = s \mid \mathrm{Pa}(Y_i) = c) = \frac{N(s, c) + \alpha}{N(c) + \alpha\,k_i},
```

where ``N(s, c)`` counts the training rows with ``Y_i = s`` and ``\mathrm{Pa}(Y_i) = c``, ``N(c)`` those with ``\mathrm{Pa}(Y_i) = c``, ``k_i`` is the number of states of ``Y_i``, and `alpha` the pseudo-count (`alpha = 0` is pure maximum likelihood; `alpha > 0` smooths, keeping never-observed states off exactly `0`). 
A parent configuration that never appears in the data falls back to a uniform distribution.

```@example parameters_learning
learned = learn_parameters_mle(dag, df; alpha = 1)
order!(learned)
```

### Expectation-Maximization

When some entries are `missing`, [`learn_parameters_em`](@ref) estimates the CPTs iteratively by Expectation-Maximization [dempster_maximum_1977](@cite). Starting from uniform tables, each iteration performs:

- an **E-step**: every row with missing values is expanded into all completions of its missing variables, each weighted by ``P(\text{missing} \mid \text{observed})`` under the current network; fully-observed rows keep weight `1`;
- an **M-step**: the CPTs are re-estimated by the same counting as maximum likelihood, summing these fractional weights instead of counting whole rows.

Iteration stops when no CPT entry moves by more than `tol`, or after `max_iter` steps. 
Because EM converges only to a *local* optimum, the (uniform) initialization matters. 
With no missing values it reduces exactly to [`learn_parameters_mle`](@ref).

```@example parameters_learning
learned = learn_parameters_em(dag, df_with_missing; alpha = 1, tol = 1e-6)
order!(learned)
```

Both estimators leave the input `dag` untouched, so the same structure can be relearned on different datasets or with different smoothing.
