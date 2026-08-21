using EnhancedBayesianNetworks

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
add_node!(dag, :R, parents = [:W])
add_node!(dag, :S, parents = [:W])
add_node!(dag, :G, parents = [:R, :S])

learned_bn = learn(dag, df)

dag = DirectAcyclicGraph()
add_node!(dag, :W, [:mixed])
add_node!(dag, :R, parents = [:W])
add_node!(dag, :S, parents = [:W])
add_node!(dag, :G, parents = [:R, :S])

learned_bn = learn(dag, df)
learned_bn.nodes[1]

learned_bn = learn(dag, df, alpha = 1.0)
learned_bn.nodes[1]

df_missing = copy(df)
df_missing.R = collect(Union{Missing, Symbol}, df_missing.R)
df_missing.R[1:200] .= missing

learned_bn = learn(dag, df_missing)
learned_bn.nodes[2]

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
