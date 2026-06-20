@testitem "Sorting - added & deleted edges" begin
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2, 3, 4]),
        Set([1]),
        Set([1]),
        Set([1])
    ])
    @test EnhancedBayesianNetworks.deleted_edges(ig, 1) == 3
    @test EnhancedBayesianNetworks.added_edges(ig, 1) == 3

    # 1 missing edge
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2, 3, 4]),
        Set([1, 3]),
        Set([1, 2]),
        Set([1])
    ])
    @test EnhancedBayesianNetworks.deleted_edges(ig, 1) == 3
    @test EnhancedBayesianNetworks.added_edges(ig, 1) == 2

    # already a clique
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2, 3, 4]),
        Set([1, 3, 4]),
        Set([1, 2, 4]),
        Set([1, 2, 3])
    ])
    @test EnhancedBayesianNetworks.deleted_edges(ig, 1) == 3
    @test EnhancedBayesianNetworks.added_edges(ig, 1) == 0

    # isolated node
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set{Int}(),
        Set{Int}()
    ])
    @test EnhancedBayesianNetworks.deleted_edges(ig, 1) == 0
    @test EnhancedBayesianNetworks.added_edges(ig, 1) == 0

    # single neighbor
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2]),
        Set([1])
    ])
    @test EnhancedBayesianNetworks.deleted_edges(ig, 1) == 1
    @test EnhancedBayesianNetworks.added_edges(ig, 1) == 0
end

@testitem "Sorting - eliminate!" begin
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2]),
        Set([1, 3]),
        Set([2])
    ])
    EnhancedBayesianNetworks.eliminate!(ig, 2)
    @test ig.neighbors[1] == Set([3])
    @test ig.neighbors[2] == Set()
    @test ig.neighbors[3] == Set([1])

    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2, 3, 4]),
        Set([1]),
        Set([1]),
        Set([1])
    ])
    EnhancedBayesianNetworks.eliminate!(ig, 1)
    @test ig.neighbors[1] == Set()
    @test ig.neighbors[2] == Set([3, 4])
    @test ig.neighbors[3] == Set([2, 4])
    @test ig.neighbors[4] == Set([2, 3])

    # already a clique
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([2, 3, 4]),
        Set([1, 3, 4]),
        Set([1, 2, 4]),
        Set([1, 2, 3])
    ])
    EnhancedBayesianNetworks.eliminate!(ig, 1)
    @test ig.neighbors[1] == Set()
    @test ig.neighbors[2] == Set([3, 4])
    @test ig.neighbors[3] == Set([2, 4])
    @test ig.neighbors[4] == Set([2, 3])

    # isolated node
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set{Int}(),
        Set([3]),
        Set([2])
    ])
    EnhancedBayesianNetworks.eliminate!(ig, 1)
    @test ig.neighbors[1] == Set()
    @test ig.neighbors[2] == Set([3])
    @test ig.neighbors[3] == Set([2])

    # Symmetry invariant
    for i in eachindex(ig.neighbors)
        for j in ig.neighbors[i]
            @test i in ig.neighbors[j]
        end
    end
end

@testitem "Sorting - complexity-score" begin
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
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([3])
        Set([5, 4])
        Set([4, 6, 1])
        Set([6, 2, 3])
        Set([6, 7, 2])
        Set([5, 4, 7, 8, 3])
        Set([5, 6])
        Set([6])
    ])
    scores = Dict(
        ns.idx_to_node[i] =>
            EnhancedBayesianNetworks.complexity_score(ig, ns, i)
        for i in eachindex(ns.idx_to_node)
    )

    @test scores[:V] == 4
    @test scores[:X] == 4
    @test scores[:S] == 8
    @test scores[:D] == 8
    @test scores[:T] == 16
    @test scores[:L] == 16
    @test scores[:B] == 16
    @test scores[:E] == 64
end

@testitem "Sorting - ic-score" begin
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
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([3]),
        Set([5, 4]),
        Set([4, 6, 1]),
        Set([6, 2, 3]),
        Set([6, 7, 2]),
        Set([5, 4, 7, 8, 3]),
        Set([5, 6]),
        Set([6])
    ])

    scores = Dict(
        i => EnhancedBayesianNetworks.ic_score(ig, ns, i)
        for i in 1:8
    )

    @test scores[1] ≈ 0.0
    @test scores[8] ≈ 0.0
    @test scores[7] ≈ 0.0
    @test scores[2] ≈ 0.5
    @test scores[3] ≈ 2/3
    @test scores[4] ≈ 2/3
    @test scores[5] ≈ 2/3
    @test scores[6] ≈ 8/5

    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set{Int}()
    ])
    @test EnhancedBayesianNetworks.ic_score(ig, ns, 1) == 0.0
end

@testitem "Sorting - ic_complexity-score" begin
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
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([3])
        Set([5, 4])
        Set([4, 6, 1])
        Set([6, 2, 3])
        Set([6, 7, 2])
        Set([5, 4, 7, 8, 3])
        Set([5, 6])
        Set([6])
    ])
    scores = Dict(
        ns.idx_to_node[i] =>
            EnhancedBayesianNetworks.ic_complexity_score(ig, ns, i)
        for i in eachindex(ns.idx_to_node)
    )
    expected = Dict(
        :T => (2/3, 16, 3),
        :D => (0.0, 8, 7),
        :L => (2/3, 16, 4),
        :V => (0.0, 4, 1),
        :S => (0.5, 8, 2),
        :B => (2/3, 16, 5),
        :X => (0.0, 4, 8),
        :E => (1.6, 64, 6),
    )

    for (node, (score, complexity, idx)) in expected
        @test scores[node][1] ≈ score
        @test scores[node][2] == complexity
        @test scores[node][3] == idx
    end
end

@testitem "Sorting - best node" begin
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
    ig = EnhancedBayesianNetworks.InteractionGraph([
        Set([3])
        Set([5, 4])
        Set([4, 6, 1])
        Set([6, 2, 3])
        Set([6, 7, 2])
        Set([5, 4, 7, 8, 3])
        Set([5, 6])
        Set([6])
    ])

    # ic
    remaining = Set(1:8)
    scorefun = EnhancedBayesianNetworks.ic_score
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 1

    remaining = Set([7, 8])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 7

    remaining = Set([6])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 6

    remaining = Set([3, 4, 6])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 3

    # complexity
    remaining = Set(1:8)
    scorefun = EnhancedBayesianNetworks.complexity_score
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 1

    remaining = Set([7, 8])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 8

    remaining = Set([6])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 6

    remaining = Set([3, 4, 6])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 3

    # ic-complexity
    remaining = Set(1:8)
    scorefun = EnhancedBayesianNetworks.ic_complexity_score
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 1

    remaining = Set([7, 8])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 8

    remaining = Set([6])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 6

    remaining = Set([3, 4, 6])
    node = EnhancedBayesianNetworks.best_node(
        ig,
        ns,
        remaining,
        scorefun
    )
    @test node == 3
end

@testitem "Sorting - sort_nodes" begin

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

    @test EnhancedBayesianNetworks.sort_nodes(bn, EnhancedBayesianNetworks.ic_score) == [1, 7, 8, 3, 2, 4, 5, 6]
    @test EnhancedBayesianNetworks.sort_nodes(bn, EnhancedBayesianNetworks.complexity_score) == [1, 8, 2, 3, 4, 5, 6, 7]
    @test EnhancedBayesianNetworks.sort_nodes(bn, EnhancedBayesianNetworks.ic_complexity_score) == [1, 8, 3, 7, 2, 4, 5, 6]
end