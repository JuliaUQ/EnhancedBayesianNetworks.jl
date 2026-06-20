@testitem "Inference - NetworkSchema" begin

    using Suppressor
    weather = DiscreteNode(:W)
    weather[:W=>:Cloudy] = 0.5
    weather[:W=>:Sunny] = 0.5
    rain = DiscreteNode(:R, [:W])
    rain[:W=>:Cloudy, :R=>:Yes] = 0.8
    rain[:W=>:Cloudy, :R=>:No] = 0.2
    rain[:W=>:Sunny, :R=>:Yes] = 0.1
    rain[:W=>:Sunny, :R=>:No] = 0.9
    sprinkler = DiscreteNode(:S, [:W])
    sprinkler[:W=>:Cloudy, :S=>:On] = 0.4
    sprinkler[:W=>:Cloudy, :S=>:Off] = 0.4
    sprinkler[:W=>:Cloudy, :S=>:broken] = 0.2
    sprinkler[:W=>:Sunny, :S=>:On] = 0.6
    sprinkler[:W=>:Sunny, :S=>:Off] = 0.3
    sprinkler[:W=>:Sunny, :S=>:broken] = 0.1
    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:No, :S=>:On, :G=>:Dry] = 0.2
    grass[:R=>:No, :S=>:On, :G=>:Wet] = 0.8
    grass[:R=>:No, :S=>:Off, :G=>:Wet] = 0.2
    grass[:R=>:No, :S=>:Off, :G=>:Dry] = 0.8
    grass[:R=>:No, :S=>:broken, :G=>:Wet] = 0.1
    grass[:R=>:No, :S=>:broken, :G=>:Dry] = 0.9
    grass[:R=>:Yes, :S=>:On, :G=>:Wet] = 0.6
    grass[:R=>:Yes, :S=>:On, :G=>:Dry] = 0.4
    grass[:R=>:Yes, :S=>:Off, :G=>:Wet] = 0.55
    grass[:R=>:Yes, :S=>:Off, :G=>:Dry] = 0.45
    grass[:R=>:Yes, :S=>:broken, :G=>:Wet] = 0.58
    grass[:R=>:Yes, :S=>:broken, :G=>:Dry] = 0.42

    nodes = [weather, rain, sprinkler, grass]
    bn = BayesianNetwork(nodes)
    add_child!(bn, :W, :R)
    add_child!(bn, :W, :S)
    add_child!(bn, :R, :G)
    add_child!(bn, :S, :G)
    @suppress order!(bn)

    idx_to_node = [:W, :R, :S, :G]
    idx_to_state = [[:Cloudy, :Sunny], [:Yes, :No], [:On, :Off, :broken], [:Dry, :Wet]]
    node_to_idx = Dict(:R => 2, :G => 4, :W => 1, :S => 3)
    state_to_idx = [Dict(:Cloudy => 1, :Sunny => 2), Dict(:No => 2, :Yes => 1), Dict(:broken => 3, :Off => 2, :On => 1), Dict(:Dry => 1, :Wet => 2)]

    ns1 = EnhancedBayesianNetworks.NetworkSchema(node_to_idx, idx_to_node, state_to_idx, idx_to_state)
    ns2 = EnhancedBayesianNetworks.NetworkSchema(bn)

    @test ns1.state_to_idx == state_to_idx
    @test ns1.idx_to_state == idx_to_state
    @test ns1.node_to_idx == node_to_idx
    @test ns1.idx_to_node == idx_to_node

    @test ns2.state_to_idx == state_to_idx
    @test ns2.idx_to_state == idx_to_state
    @test ns2.node_to_idx == node_to_idx
    @test ns2.idx_to_node == idx_to_node
end

@testitem "Inference - InteractionGraph" begin

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
    ig = EnhancedBayesianNetworks.InteractionGraph(bn)
    @test ig.neighbors[1] == Set([3])
    @test ig.neighbors[2] == Set([5, 4])
    @test ig.neighbors[3] == Set([4, 6, 1])
    @test ig.neighbors[4] == Set([6, 2, 3])
    @test ig.neighbors[5] == Set([6, 7, 2])
    @test ig.neighbors[6] == Set([5, 4, 7, 8, 3])
    @test ig.neighbors[7] == Set([5, 6])
    @test ig.neighbors[8] == Set([6])
end