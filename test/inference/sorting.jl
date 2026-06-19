@testitem "Inference - ic scores" begin
    neighbors = [
        Set([3]),
        Set([5, 4]),
        Set([4, 6, 1]),
        Set([6, 2, 3]),
        Set([6, 7, 2]),
        Set([5, 4, 7, 8, 3]),
        Set([5, 6]),
        Set([6])
    ]
    ig = EnhancedBayesianNetworks.InteractionGraph(neighbors)

    @test EnhancedBayesianNetworks.deleted_edges(ig, 1) == 1
    @test EnhancedBayesianNetworks.deleted_edges(ig, 2) == 2
    @test EnhancedBayesianNetworks.deleted_edges(ig, 3) == 3
    @test EnhancedBayesianNetworks.deleted_edges(ig, 6) == 5
    @test EnhancedBayesianNetworks.deleted_edges(ig, 8) == 1

    @test EnhancedBayesianNetworks.added_edges(ig, 1) == 0
    @test EnhancedBayesianNetworks.added_edges(ig, 2) == 1
    @test EnhancedBayesianNetworks.added_edges(ig, 7) == 0
    @test EnhancedBayesianNetworks.added_edges(ig, 6) == 8

    @test EnhancedBayesianNetworks.ic_score(ig, 1) ≈ 0.0
    @test EnhancedBayesianNetworks.ic_score(ig, 2) ≈ 0.5
    @test EnhancedBayesianNetworks.ic_score(ig, 3) ≈ 2/3
    @test EnhancedBayesianNetworks.ic_score(ig, 4) ≈ 2/3
    @test EnhancedBayesianNetworks.ic_score(ig, 5) ≈ 2/3
    @test EnhancedBayesianNetworks.ic_score(ig, 6) ≈ 1.6
    @test EnhancedBayesianNetworks.ic_score(ig, 7) ≈ 0.0
    @test EnhancedBayesianNetworks.ic_score(ig, 8) ≈ 0.0

    remaining = Set(1:length(ig.neighbors))
    node = EnhancedBayesianNetworks.best_node(ig, remaining, EnhancedBayesianNetworks.ic_score)
    @test node == 1

    ig = EnhancedBayesianNetworks.InteractionGraph(neighbors)
    EnhancedBayesianNetworks.eliminate!(ig, 1)

    @test isempty(ig.neighbors[1])
    @test !(1 in ig.neighbors[3])
    @test ig.neighbors[3] == Set([4, 6])

    ig = EnhancedBayesianNetworks.InteractionGraph(neighbors)

    @test !(5 in ig.neighbors[4])
    @test !(4 in ig.neighbors[5])

    EnhancedBayesianNetworks.eliminate!(ig, 2)

    @test 5 in ig.neighbors[4]
    @test 4 in ig.neighbors[5]
    @test isempty(ig.neighbors[2])
    @test !(2 in ig.neighbors[4])
    @test !(2 in ig.neighbors[5])
    @test EnhancedBayesianNetworks.deleted_edges(ig, 4) == 3

    ig = EnhancedBayesianNetworks.InteractionGraph(neighbors)
    EnhancedBayesianNetworks.eliminate!(ig, 6)
    @test length(ig.neighbors[3]) == 4
    @test length(ig.neighbors[4]) == 4
    @test length(ig.neighbors[5]) == 4
    @test length(ig.neighbors[7]) == 4
    @test length(ig.neighbors[8]) == 4

    neigh = [3, 4, 5, 7, 8]

    for i in eachindex(neigh)
        for j in (i+1):length(neigh)
            @test neigh[j] in ig.neighbors[neigh[i]]
            @test neigh[i] in ig.neighbors[neigh[j]]
        end
    end
    for i in eachindex(ig.neighbors)
        for j in ig.neighbors[i]
            @test i in ig.neighbors[j]
        end
    end

    ig = EnhancedBayesianNetworks.InteractionGraph(neighbors)
    EnhancedBayesianNetworks.eliminate!(ig, 2)
    old = deepcopy(ig)
    EnhancedBayesianNetworks.eliminate!(ig, 2)

    @test ig.neighbors == old.neighbors
end

@testitem "Inference - complexity scores" begin
    neighbors = [
        Set([3]),
        Set([5, 4]),
        Set([4, 6, 1]),
        Set([6, 2, 3]),
        Set([6, 7, 2]),
        Set([5, 4, 7, 8, 3]),
        Set([5, 6]),
        Set([6])
    ]
    ig = EnhancedBayesianNetworks.InteractionGraph(neighbors)

    idx_to_node = [:V, :S, :T, :L, :B, :E, :D, :X]
    idx_to_state = [
        [:YesV, :NoV],
        [:YesS, :NoS],
        [:YesT, :NoT],
        [:YesL, :NoL],
        [:YesB, :NoB],
        [:YesE, :NoE],
        [:YesD, :NoD],
        [:YesX, :NoX]
    ]
    node_to_idx = Dict(:T => 3, :D => 7, :L => 4, :V => 1, :S => 2, :B => 5, :X => 8, :E => 6)
    state_to_idx = [
        Dict(:YesV => 1, :NoV => 2),
        Dict(:NoS => 2, :YesS => 1),
        Dict(:YesT => 1, :NoT => 2),
        Dict(:YesL => 1, :NoL => 2),
        Dict(:YesB => 1, :NoB => 2),
        Dict(:YesE => 1, :NoE => 2),
        Dict(:YesD => 1, :NoD => 2),
        Dict(:YesX => 1, :NoX => 2)
    ]
    ns = EnhancedBayesianNetworks.NetworkSchema(node_to_idx, idx_to_node, state_to_idx, idx_to_state)

    @test EnhancedBayesianNetworks.complexity_score(ig, 1, ns) == 4
    @test EnhancedBayesianNetworks.complexity_score(ig, 2, ns) == 8
    @test EnhancedBayesianNetworks.complexity_score(ig, 3, ns) == 16
    @test EnhancedBayesianNetworks.complexity_score(ig, 4, ns) == 16
    @test EnhancedBayesianNetworks.complexity_score(ig, 5, ns) == 16
    @test EnhancedBayesianNetworks.complexity_score(ig, 6, ns) == 64
    @test EnhancedBayesianNetworks.complexity_score(ig, 7, ns) == 8
    @test EnhancedBayesianNetworks.complexity_score(ig, 8, ns) == 4

    remaining = Set(1:8)
    node = EnhancedBayesianNetworks.best_node(
        ig,
        remaining,
        (ig, node) -> EnhancedBayesianNetworks.complexity_score(ig, node, ns)
    )
    @test node == 1

    ig2 = deepcopy(ig)
    EnhancedBayesianNetworks.eliminate!(ig2, 1)
    remaining = Set(2:8)
    node = EnhancedBayesianNetworks.best_node(
        ig2,
        remaining,
        (ig, node) -> EnhancedBayesianNetworks.complexity_score(ig, node, ns)
    )
    @test node == 8
end

@testitem "Inference - sorting" begin

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

    # IC
    order = EnhancedBayesianNetworks.sort_with_minimal_added_complexity(bn)
    @test length(order) == length(bn.nodes)
    @test sort(order) == collect(1:length(bn.nodes))
    ig = EnhancedBayesianNetworks.InteractionGraph(bn)

    scores = [EnhancedBayesianNetworks.ic_score(ig, i) for i in 1:length(ig.neighbors)]
    mins = findall(==(minimum(scores)), scores)
    @test order[1] in mins

    ig = EnhancedBayesianNetworks.InteractionGraph(bn)
    firstnode = order[1]
    EnhancedBayesianNetworks.eliminate!(ig, firstnode)
    remaining = Set(setdiff(1:8, [firstnode]))
    secondnode = EnhancedBayesianNetworks.best_node(ig, remaining, EnhancedBayesianNetworks.ic_score)
    @test order[2] == secondnode

    ig = EnhancedBayesianNetworks.InteractionGraph(bn)
    remaining = Set(1:8)
    expected = Int[]

    while !isempty(remaining)
        node = EnhancedBayesianNetworks.best_node(ig, remaining, EnhancedBayesianNetworks.ic_score)
        push!(expected, node)
        delete!(remaining, node)
        EnhancedBayesianNetworks.eliminate!(ig, node)
    end
    @test order == expected

    # complexity
    ns = EnhancedBayesianNetworks.NetworkSchema(bn)
    order = EnhancedBayesianNetworks.sort_with_minimal_complexity(bn, ns)
    @test length(order) == 8
    @test sort(order) == collect(1:8)

    ig = EnhancedBayesianNetworks.InteractionGraph(bn)
    remaining = Set(1:8)
    expected = Int[]

    while !isempty(remaining)
        node = EnhancedBayesianNetworks.best_node(
            ig,
            remaining,
            (ig, node) -> EnhancedBayesianNetworks.complexity_score(ig, node, ns)
        )
        push!(expected, node)
        delete!(remaining, node)
        EnhancedBayesianNetworks.eliminate!(ig, node)
    end
    @test order == expected

    # Both complexities
    order = EnhancedBayesianNetworks.sort_with_minimal_added_complexity_and_complexity(bn, ns)
    @test length(order) == 8
    @test sort(order) == collect(1:8)

    ig = EnhancedBayesianNetworks.InteractionGraph(bn)
    remaining = Set(1:8)
    expected = Int[]
    scorefun =
        (ig, node) -> (EnhancedBayesianNetworks.ic_score(ig, node), EnhancedBayesianNetworks.complexity_score(ig, node, ns), node)

    while !isempty(remaining)
        node = EnhancedBayesianNetworks.best_node(ig, remaining, scorefun)
        push!(expected, node)
        delete!(remaining, node)
        EnhancedBayesianNetworks.eliminate!(ig, node)
    end

    @test order == expected
end

@testitem "Inference - sort_node" begin

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
    order = EnhancedBayesianNetworks.sort_nodes(bn, EnhancedBayesianNetworks.ic_score)

    @test length(order) == 8
    @test sort(order) == collect(1:8)
end