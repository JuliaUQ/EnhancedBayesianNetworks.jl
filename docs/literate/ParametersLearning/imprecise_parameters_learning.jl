# # Imprecise Parameters Learning
#
# The [Precise Parameters Learning](precise_parameters_learning.md) example estimates every CPT entry as a single number, so a learned `0.5` looks the same whether a thousand rows support it or none at all. Imprecise estimators report an `Interval` instead, and the learned network is a [`CredalNetwork`](@ref). This example learns the same network from the same data with [`ImpreciseDirichletModel`](@ref) and [`QuesenberryHurst`](@ref), shows what their intervals do and do not promise, bounds the estimates over every completion of incomplete data, and keeps credal inference feasible with per-node estimators. The theory behind each estimator is in the [Parameter Learning](../../manual/parameterlearning.md) chapter of the manual.

using EnhancedBayesianNetworks
using DataFrames
using Random

Random.seed!(1)    # the same seed as the precise example, so the data are the same too
nothing # hide

# ## The reference network
#
# The sprinkler network of the precise example: the weather drives whether it rains and whether the sprinkler runs, and both wet the grass. It is used only to *generate* data.

W = DiscreteNode(:W)
W[:W => :sunny] = 0.7
W[:W => :cloudy] = 0.3

R = DiscreteNode(:R, [:W])
R[:W => :sunny, :R => :yesR] = 0.1
R[:W => :sunny, :R => :noR] = 0.9
R[:W => :cloudy, :R => :yesR] = 0.8
R[:W => :cloudy, :R => :noR] = 0.2

S = DiscreteNode(:S, [:W])
S[:W => :sunny, :S => :onS] = 0.5
S[:W => :sunny, :S => :offS] = 0.5
S[:W => :cloudy, :S => :onS] = 0.1
S[:W => :cloudy, :S => :offS] = 0.9

G = DiscreteNode(:G, [:R, :S])
G[:R => :yesR, :S => :onS, :G => :wetG] = 0.99
G[:R => :yesR, :S => :onS, :G => :dryG] = 0.01
G[:R => :yesR, :S => :offS, :G => :wetG] = 0.9
G[:R => :yesR, :S => :offS, :G => :dryG] = 0.1
G[:R => :noR, :S => :onS, :G => :wetG] = 0.9
G[:R => :noR, :S => :onS, :G => :dryG] = 0.1
G[:R => :noR, :S => :offS, :G => :wetG] = 0.0
G[:R => :noR, :S => :offS, :G => :dryG] = 1.0

bn = BayesianNetwork([W, R, S, G])
add_child!(bn, :W, :R)
add_child!(bn, :W, :S)
add_child!(bn, :R, :G)
add_child!(bn, :S, :G)
order!(bn)

df = sample(bn, 1000)

dag = DirectAcyclicGraph()
add_node!(dag, :W)
add_node!(dag, :R; parents = [:W])
add_node!(dag, :S; parents = [:W])
add_node!(dag, :G; parents = [:R, :S])
nothing # hide

# ## Imprecise Dirichlet model
#
# [`ImpreciseDirichletModel`](@ref)`()`, with the default prior strength `s = 1`, gives each entry the interval of posterior means over every Dirichlet prior of that strength. A precise entry is printed as a number, an imprecise one as an interval:

learned_idm = learn(dag, df, ImpreciseDirichletModel())
order!(learned_idm)
learned_idm.nodes[learned_idm.topology[:R]]

# To compare a learned credal network with the true one, we check node by node whether every true CPT entry lies inside its learned interval, and how wide the widest interval is. As in the precise example, the CPTs are joined on their state columns.

function interval_check(learned, truth)
    return DataFrame(
        map(truth.nodes) do node
            t = node.cpt.data
            l = learned.nodes[learned.topology[node.name]].cpt.data
            joined = innerjoin(t, l; on = setdiff(propertynames(t), [:Π]), makeunique = true)
            lower = [p isa Interval ? p.lb : p for p in joined.Π_1]
            upper = [p isa Interval ? p.ub : p for p in joined.Π_1]
            (node = node.name, contains_truth = all(lower .<= joined.Π .<= upper), max_width = maximum(upper .- lower))
        end
    )
end

interval_check(learned_idm, bn)

# The intervals are narrow, and they **miss** the true values. That is not a failure of the estimator but a statement of what it measures: an imprecise Dirichlet interval is the range of estimates over a set of *priors*, and has width `s / (n + s)` for a configuration seen `n` times. With a thousand rows the prior hardly matters, so the interval is tight around the relative frequency, whatever the sampling error of that frequency. It says how much the conclusion depends on the prior, not where the true value lies.

# ## Quesenberry–Hurst confidence intervals
#
# [`QuesenberryHurst`](@ref)`()` answers the other question. Its intervals are simultaneous 95% confidence intervals for the proportions of each column: over repeated samples, all the entries of a column are covered at once in 95% of them.

learned_qh = learn(dag, df, QuesenberryHurst())
order!(learned_qh)
learned_qh.nodes[learned_qh.topology[:R]]

#-

interval_check(learned_qh, bn)

# The intervals are wider, and contain the truth for every node but `G`. The miss is the column with no rain and the sprinkler on: the true probability of dry grass is `0.1`, but this sample has only 18 dry days out of 327, a sampling fluctuation the 95% interval does not reach. A 95% guarantee per column means that about one column in twenty is expected to miss, and this network has nine.

learned_qh.nodes[learned_qh.topology[:G]]

# The two kinds of interval also shrink at different rates as data accumulate: an imprecise Dirichlet interval as one over the sample size, a confidence interval as one over its square root, the rate of the sampling error itself.

DataFrame(
    map([100, 1_000, 10_000]) do n
        data = sample(bn, n)
        (
            rows = n,
            idm_width = maximum(interval_check(learn(dag, data, ImpreciseDirichletModel()), bn).max_width),
            qh_width = maximum(interval_check(learn(dag, data, QuesenberryHurst()), bn).max_width),
        )
    end
)

# Both are valid credal networks, but they answer different questions, and a result should say which one produced it.

# ## Configurations without data
#
# Where the two families of estimators differ most from a precise one is where there are no data at all. Declaring a third weather state, `:foggy`, that never occurs in the sample:

dag_foggy = DirectAcyclicGraph()
add_node!(dag_foggy, :W; states = [:foggy])
add_node!(dag_foggy, :R; parents = [:W])
add_node!(dag_foggy, :S; parents = [:W])
add_node!(dag_foggy, :G; parents = [:R, :S])

learned_foggy = learn(dag_foggy, df, ImpreciseDirichletModel())
learned_foggy.nodes[learned_foggy.topology[:R]]

# The rain distribution on a foggy day is the **vacuous** interval `[0, 1]`: the data say nothing about it, and the network says so, where a precise estimator would have reported the uniform `0.5` as if it had been measured.

# ## Incomplete data: bounds over every completion
#
# As in the precise example, we delete 30% of the rain observations at random:

df_missing = allowmissing(df)
df_missing.R[randperm(nrow(df))[1:300]] .= missing
nothing # hide

# With an imprecise estimator, [`learn`](@ref) assumes nothing about why the values are missing. Every entry is bounded over all the completions of the data consistent with what was observed, so the intervals of the rain node widen by exactly as much as the 300 unknown values could matter:

learned_missing = learn(dag, df_missing, ImpreciseDirichletModel())
order!(learned_missing)
learned_missing.nodes[learned_missing.topology[:R]]

#-

interval_check(learned_missing, bn)

# The rain intervals are now wide enough to contain the true values, since the missing values, not the prior, are what limits the estimate. The other nodes are unaffected: every row still records the weather, the sprinkler and the grass. For comparison, expectation maximisation, which assumes the values are missing at random, gives single numbers that here lie inside those bounds:

learned_em = learn(dag, df_missing, PreciseDirichletModel())
learned_em.nodes[learned_em.topology[:R]]

# [`QuesenberryHurst`](@ref) cannot be fitted from incomplete data, since its confidence region has no closed-form envelope over the completions, and [`learn`](@ref) refuses it:

try
    learn(dag, df_missing, QuesenberryHurst())
catch e
    println(e.msg)
end

# ## Inference on a learned credal network
#
# A learned credal network is queried with [`infer`](@ref) like any other, and returns bounds. Asking again for the probability of rain given that the grass is wet:

infer(learned_idm, :R, Evidence(:G => :wetG))

#-

infer(learned_missing, :R, Evidence(:G => :wetG))

# The true probability of rain is `0.496`. The complete-data bounds are narrow and, like the CPT intervals they come from, do not contain it; the bounds from the incomplete data are wide and do. In both cases [`infer`](@ref) visited 512 **extreme networks**, one for every combination of a vertex of each column's credal set: here 9 columns with 2 vertices each.

# ## Keeping credal inference feasible
#
# The number of extreme networks is a product over every column of every imprecise node, so it grows exponentially with the size of the network. [`infer`](@ref) counts them before building any, and refuses a network with more than `max_networks`, which defaults to `10^6`:

try
    infer(learned_idm, :R, Evidence(:G => :wetG); max_networks = 100)
catch e
    println(e.msg)
end

# The remedy is to be imprecise only where the data are thin. With per-node estimators, declared through the `estimator` keyword of [`add_node!`](@ref), a DAG can learn most nodes precisely and only a few as intervals. On a small sample of thirty rows, the grass node is the one whose rare configurations are poorly observed, so it alone is learned imprecisely:

small = sample(bn, 30)

mixed = DirectAcyclicGraph()
add_node!(mixed, :W; estimator = PreciseDirichletModel(1))
add_node!(mixed, :R; parents = [:W], estimator = PreciseDirichletModel(1))
add_node!(mixed, :S; parents = [:W], estimator = PreciseDirichletModel(1))
add_node!(mixed, :G; parents = [:R, :S], estimator = ImpreciseDirichletModel())

learned_mixed = learn(mixed, small)
order!(learned_mixed)
infer(learned_mixed, :R, Evidence(:G => :wetG))

# Only the four columns of the grass node are imprecise, so inference visits 16 extreme networks instead of 512, and the bounds show how much the answer depends on the poorly observed part of the network alone.
