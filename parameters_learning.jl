using EnhancedBayesianNetworks

V = DiscreteNode(:V)
V[:V=>:YesV] = 0.01
V[:V=>:NoV] = 0.99

S = DiscreteNode(:S)
S[:S=>:YesS] = 0.01
S[:S=>:NoS] = 0.99

T = DiscreteNode(:T, [:V])
T[:V=>:YesV, :T=>:YesT] = 0.05
T[:V=>:YesV, :T=>:NoT] = 0.95
T[:V=>:NoV, :T=>:YesT] = 0.01
T[:V=>:NoV, :T=>:NoT] = 0.99

L = DiscreteNode(:L, [:S])
L[:S=>:YesS, :L=>:YesL] = 0.1
L[:S=>:YesS, :L=>:NoL] = 0.9
L[:S=>:NoS, :L=>:YesL] = 0.01
L[:S=>:NoS, :L=>:NoL] = 0.99

B = DiscreteNode(:B, [:S])
B[:S=>:YesS, :B=>:YesB] = 0.6
B[:S=>:YesS, :B=>:NoB] = 0.4
B[:S=>:NoS, :B=>:YesB] = 0.3
B[:S=>:NoS, :B=>:NoB] = 0.7

E = DiscreteNode(:E, [:L, :T])
E[:L=>:YesL, :T=>:YesT, :E=>:YesE] = 1
E[:L=>:YesL, :T=>:YesT, :E=>:NoE] = 0
E[:L=>:YesL, :T=>:NoT, :E=>:YesE] = 1
E[:L=>:YesL, :T=>:NoT, :E=>:NoE] = 0
E[:L=>:NoL, :T=>:YesT, :E=>:YesE] = 1
E[:L=>:NoL, :T=>:YesT, :E=>:NoE] = 0
E[:L=>:NoL, :T=>:NoT, :E=>:YesE] = 0
E[:L=>:NoL, :T=>:NoT, :E=>:NoE] = 1

D = DiscreteNode(:D, [:B, :E])
D[:B=>:YesB, :E=>:YesE, :D=>:YesD] = 0.9
D[:B=>:YesB, :E=>:YesE, :D=>:NoD] = 0.1
D[:B=>:YesB, :E=>:NoE, :D=>:YesD] = 0.8
D[:B=>:YesB, :E=>:NoE, :D=>:NoD] = 0.2
D[:B=>:NoB, :E=>:YesE, :D=>:YesD] = 0.7
D[:B=>:NoB, :E=>:YesE, :D=>:NoD] = 0.3
D[:B=>:NoB, :E=>:NoE, :D=>:YesD] = 0.1
D[:B=>:NoB, :E=>:NoE, :D=>:NoD] = 0.9

X = DiscreteNode(:X, [:E])
X[:E=>:YesE, :X=>:YesX] = 0.98
X[:E=>:YesE, :X=>:NoX] = 0.02
X[:E=>:NoE, :X=>:YesX] = 0.05
X[:E=>:NoE, :X=>:NoX] = 0.95

nodes = [V, S, T, L, B, E, D, X]
bn = BayesianNetwork(nodes)
add_child!(bn, V, T)
add_child!(bn, S, [L, B])
add_child!(bn, [T, L], E)
add_child!(bn, [E, B], D)
add_child!(bn, E, X)
order!(bn)


df = sample(bn, 100)



dag = DirectAcyclicGraph()
add_node!(dag, :V, [:maybe])
add_node!(dag, :S)
add_node!(dag, :T, parents=[:V])
add_node!(dag, :L, parents=[:S])
add_node!(dag, :B, parents=[:S])
add_node!(dag, :E, parents=[:L, :T])
add_node!(dag, :D, parents=[:B, :E])
add_node!(dag, :X, parents=[:E])
