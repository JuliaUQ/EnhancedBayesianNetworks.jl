using EnhancedBayesianNetworks
using DataFrames
using Random

Random.seed!(1)    # fixed seed, so that the samples and the numbers below are reproducible
nothing # hide

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

df = sample(bn, 1000)
first(df, 10)

dag = DirectAcyclicGraph()
add_node!(dag, :W)
add_node!(dag, :R; parents = [:W])
add_node!(dag, :S; parents = [:W])
add_node!(dag, :G; parents = [:R, :S])
nothing # hide

learned = learn(dag, df, PreciseDirichletModel())
order!(learned)
learned.nodes[learned.topology[:R]]

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

DataFrame(
    map([100, 1_000, 10_000]) do n
        errors = cpt_errors(learn(dag, sample(bn, n), PreciseDirichletModel()), bn)
        (rows = n, max_error = maximum(errors.max_error))
    end
)

small = sample(bn, 30)
learned_small = learn(dag, small, PreciseDirichletModel())
learned_small.nodes[learned_small.topology[:G]]

learned_laplace = learn(dag, small, PreciseDirichletModel(1))
learned_laplace.nodes[learned_laplace.topology[:G]]

dag_foggy = DirectAcyclicGraph()
add_node!(dag_foggy, :W; states = [:foggy])
add_node!(dag_foggy, :R; parents = [:W])
add_node!(dag_foggy, :S; parents = [:W])
add_node!(dag_foggy, :G; parents = [:R, :S])
nothing # hide

learned_foggy = learn(dag_foggy, df, PreciseDirichletModel())
learned_foggy.nodes[learned_foggy.topology[:W]]

learned_foggy.nodes[learned_foggy.topology[:R]]

learned_foggy = learn(dag_foggy, df, PreciseDirichletModel(1))
learned_foggy.nodes[learned_foggy.topology[:W]]

df_missing = allowmissing(df)
df_missing.R[randperm(nrow(df))[1:300]] .= missing
first(df_missing, 10)

learned_em = learn(dag, df_missing, PreciseDirichletModel())
order!(learned_em)
cpt_errors(learned_em, bn)

cpt_errors(learn(dag, dropmissing(df_missing), PreciseDirichletModel()), bn)

infer(learned, :R, Evidence(:G => :wetG))

infer(learned_em, :R, Evidence(:G => :wetG))

infer(bn, :R, Evidence(:G => :wetG))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
