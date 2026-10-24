# # Precise Parameters Learning
#
# The networks in the other examples are written out by hand, with every conditional probability given. Often the structure is known but the numbers are not: they have to be **learned from data**. This example estimates the CPTs of a [`BayesianNetwork`](@ref) with [`PreciseDirichletModel`](@ref): by maximum likelihood and by Laplace smoothing on complete data, and by expectation maximisation when some observations are missing. The theory behind each estimator is in the [Parameter Learning](../../manual/parameterlearning.md) chapter of the manual.
#
# To have a ground truth to compare against, we first build a fully specified network, draw synthetic data from it, and then try to recover its parameters.

using EnhancedBayesianNetworks
using DataFrames
using Random

Random.seed!(1)    # fixed seed, so that the samples and the numbers below are reproducible
nothing # hide

# ## The reference network
#
# The familiar sprinkler network: the weather drives whether it rains and whether the sprinkler runs, and both wet the grass. It is used only to *generate* data; the learning step never sees these probabilities.

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

gplot(bn, background_color = "white")

# ## Generating a dataset
#
# [`sample`](@ref) draws independent realizations of the whole network. A thousand rows form our observed dataset, one column per node and one row per draw; only the first ten are shown.

df = sample(bn, 1000)
first(df, 10)

# ## Declaring the structure
#
# Learning needs a structure to fill: a [`DirectAcyclicGraph`](@ref), built node by node with [`add_node!`](@ref), parents first. It holds no probabilities at all.

dag = DirectAcyclicGraph()
add_node!(dag, :W)
add_node!(dag, :R; parents = [:W])
add_node!(dag, :S; parents = [:W])
add_node!(dag, :G; parents = [:R, :S])
nothing # hide

# ## Maximum likelihood
#
# [`PreciseDirichletModel`](@ref)`()`, with the default `α = 0`, is maximum likelihood: every CPT entry is the relative frequency of the state among the rows with the same parent configuration.

learned = learn(dag, df, PreciseDirichletModel())
order!(learned)
learned.nodes[learned.topology[:R]]

# The true values are `0.1` and `0.8` for rain on a sunny and on a cloudy day. To compare the whole network at once we measure, node by node, the largest absolute difference between a learned CPT entry and the true one. The CPTs are joined on their state columns, since the learned rows need not come in the same order as the true ones.

function cpt_errors(learned, truth)
    return DataFrame(
        map(truth.nodes) do node
            t = node.cpt.data
            l = learned.nodes[learned.topology[node.name]].cpt.data
            joined = innerjoin(t, l; on = setdiff(propertynames(t), [:Π]), makeunique = true)
            (node = node.name, max_error = maximum(abs.(joined.Π .- joined.Π_1)))
        end
    )
end

cpt_errors(learned, bn)

# The error comes from sampling alone, so it shrinks as the dataset grows, roughly as one over the square root of the sample size:

DataFrame(
    map([100, 1_000, 10_000]) do n
        errors = cpt_errors(learn(dag, sample(bn, n), PreciseDirichletModel()), bn)
        (rows = n, max_error = maximum(errors.max_error))
    end
)

# ## Small samples and Laplace smoothing
#
# On a small sample some parent configurations are seen only a handful of times, and maximum likelihood takes those few rows at face value. With thirty rows, the configurations where it rains and the sprinkler is on are rare, and their entries come out as hard `0` and `1`:

small = sample(bn, 30)
learned_small = learn(dag, small, PreciseDirichletModel())
learned_small.nodes[learned_small.topology[:G]]

# [`PreciseDirichletModel`](@ref)`(1)` is **Laplace smoothing**: one pseudo-observation per state pulls every estimate towards the uniform, strongly where data are scarce and hardly at all where they are plentiful. No entry is exactly `0` or `1` any more, so the learned network never rules out an event just because the sample missed it.

learned_laplace = learn(dag, small, PreciseDirichletModel(1))
learned_laplace.nodes[learned_laplace.topology[:G]]

# ## Declaring an unobserved state
#
# A node's states are read from the data, but extra states can be **declared** with the `states` keyword of [`add_node!`](@ref) even if they never occur. Here a third weather state, `:foggy`, is possible in principle but absent from the sample.

dag_foggy = DirectAcyclicGraph()
add_node!(dag_foggy, :W; states = [:foggy])
add_node!(dag_foggy, :R; parents = [:W])
add_node!(dag_foggy, :S; parents = [:W])
add_node!(dag_foggy, :G; parents = [:R, :S])
nothing # hide

# Under maximum likelihood an unobserved state gets probability exactly zero, and the rain distribution on a foggy day, a parent configuration with no data at all, falls back to the uniform:

learned_foggy = learn(dag_foggy, df, PreciseDirichletModel())
learned_foggy.nodes[learned_foggy.topology[:W]]

#-

learned_foggy.nodes[learned_foggy.topology[:R]]

# With Laplace smoothing `:foggy` keeps a small probability instead of a hard zero. The foggy rain distribution is still uniform, and still only the prior speaking: nothing in a precise CPT tells a measured `0.5` from an assumed one. The [Imprecise Parameters Learning](imprecise_parameters_learning.md) example reports that case as the interval `[0, 1]` instead.

learned_foggy = learn(dag_foggy, df, PreciseDirichletModel(1))
learned_foggy.nodes[learned_foggy.topology[:W]]

# ## Incomplete data: expectation maximisation
#
# Real datasets have gaps. Here we delete 30% of the rain observations, chosen at random, and mark them `missing`. Because the rows to delete are picked independently of every value, the data are **missing at random**, the assumption under which expectation maximisation is valid.

df_missing = allowmissing(df)
df_missing.R[randperm(nrow(df))[1:300]] .= missing
first(df_missing, 10)

# When every node is precise, [`learn`](@ref) completes the data by expectation maximisation: each incomplete row is split between the states it could have taken, in proportion to their probability under the current estimate, and the CPTs are re-estimated from those weighted rows until no entry changes by more than `tol`.

learned_em = learn(dag, df_missing, PreciseDirichletModel())
order!(learned_em)
cpt_errors(learned_em, bn)

# For comparison, simply dropping the incomplete rows also gives unbiased estimates when values are missing at random, but it throws away the weather, sprinkler and grass observations of those 300 rows along with the missing rain:

cpt_errors(learn(dag, dropmissing(df_missing), PreciseDirichletModel()), bn)

# Expectation maximisation keeps every observed value. Its estimates are only as good as the missing-at-random assumption, though: if rain went unrecorded mostly on rainy days, both approaches would be biased, and nothing in the result would show it.

# ## Inference on the learned network
#
# A learned network is an ordinary [`BayesianNetwork`](@ref). Asking for the probability of rain given that the grass is wet, the network learned from complete data, the one learned by expectation maximisation and the true network give close answers:

infer(learned, :R, Evidence(:G => :wetG))

#-

infer(learned_em, :R, Evidence(:G => :wetG))

#-

infer(bn, :R, Evidence(:G => :wetG))
