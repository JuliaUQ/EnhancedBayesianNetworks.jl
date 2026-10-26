using EnhancedBayesianNetworks
using DataFrames
using Random

Random.seed!(1)    # the same seed as the precise example, so the data are the same too
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

df = sample(bn, 1000)

dag = DirectAcyclicGraph()
add_node!(dag, :W)
add_node!(dag, :R; parents = [:W])
add_node!(dag, :S; parents = [:W])
add_node!(dag, :G; parents = [:R, :S])
nothing # hide

learned_idm = learn(dag, df, ImpreciseDirichletModel())
order!(learned_idm)
learned_idm.nodes[learned_idm.topology[:R]]

function interval_check(learned, truth)
    return DataFrame(
        map(truth.nodes) do node
            t = node.cpt.data
            l = learned.nodes[learned.topology[node.name]].cpt.data
            joined = innerjoin(t, l; on = setdiff(propertynames(t), [:Π]), makeunique = true)
            lower = [p isa Interval ? p.lb : p for p in joined.Π_1]
            upper = [p isa Interval ? p.ub : p for p in joined.Π_1]
            distance = max.(lower .- joined.Π, joined.Π .- upper, 0.0)
            (node = node.name, contains_truth = all(iszero, distance), max_distance = maximum(distance), max_width = maximum(upper .- lower))
        end
    )
end

interval_check(learned_idm, bn)

learned_qh = learn(dag, df, QuesenberryHurst())
order!(learned_qh)
learned_qh.nodes[learned_qh.topology[:R]]

interval_check(learned_qh, bn)

learned_qh.nodes[learned_qh.topology[:G]]

DataFrame(
    map([100, 1_000, 10_000]) do n
        data = sample(bn, n)
        idm = interval_check(learn(dag, data, ImpreciseDirichletModel()), bn)
        qh = interval_check(learn(dag, data, QuesenberryHurst()), bn)
        (
            rows = n,
            idm_width = maximum(idm.max_width),
            idm_distance = maximum(idm.max_distance),
            qh_width = maximum(qh.max_width),
            qh_distance = maximum(qh.max_distance),
        )
    end
)

dag_foggy = DirectAcyclicGraph()
add_node!(dag_foggy, :W; states = [:foggy])
add_node!(dag_foggy, :R; parents = [:W])
add_node!(dag_foggy, :S; parents = [:W])
add_node!(dag_foggy, :G; parents = [:R, :S])

learned_foggy = learn(dag_foggy, df, ImpreciseDirichletModel())
learned_foggy.nodes[learned_foggy.topology[:R]]

df_missing = allowmissing(df)
df_missing.R[randperm(nrow(df))[1:300]] .= missing
nothing # hide

learned_missing = learn(dag, df_missing, ImpreciseDirichletModel())
order!(learned_missing)
learned_missing.nodes[learned_missing.topology[:R]]

interval_check(learned_missing, bn)

learned_em = learn(dag, df_missing, PreciseDirichletModel())
learned_em.nodes[learned_em.topology[:R]]

try
    learn(dag, df_missing, QuesenberryHurst())
catch e
    println(e.msg)
end

infer(learned_idm, :R, Evidence(:G => :wetG))

infer(learned_missing, :R, Evidence(:G => :wetG))

try
    infer(learned_idm, :R, Evidence(:G => :wetG); max_networks = 100)
catch e
    println(e.msg)
end

small = sample(bn, 30)

mixed = DirectAcyclicGraph()
add_node!(mixed, :W; estimator = PreciseDirichletModel(1))
add_node!(mixed, :R; parents = [:W], estimator = PreciseDirichletModel(1))
add_node!(mixed, :S; parents = [:W], estimator = PreciseDirichletModel(1))
add_node!(mixed, :G; parents = [:R, :S], estimator = ImpreciseDirichletModel())

learned_mixed = learn(mixed, small)
order!(learned_mixed)
infer(learned_mixed, :R, Evidence(:G => :wetG))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
