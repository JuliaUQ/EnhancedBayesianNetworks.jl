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

@testitem "Sorting - min complexity" begin
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
            EnhancedBayesianNetworks.factor_score(ig, ns, i)
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

@testitem "Sorting - min added complexity" begin
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
        i => EnhancedBayesianNetworks.fill_score(ig, ns, i)
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
    @test EnhancedBayesianNetworks.fill_score(ig, ns, 1) == 0.0
end

@testitem "Sorting - min added complexity & complexity" begin
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
            EnhancedBayesianNetworks.fill_factor_score(ig, ns, i)
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
    scorefun = EnhancedBayesianNetworks.fill_score
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
    scorefun = EnhancedBayesianNetworks.factor_score
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
    scorefun = EnhancedBayesianNetworks.fill_factor_score
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
    @test EnhancedBayesianNetworks.sort_nodes(ig, ns, EnhancedBayesianNetworks.fill_score) == [1, 7, 8, 3, 2, 4, 5, 6]

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
    @test EnhancedBayesianNetworks.sort_nodes(ig, ns, EnhancedBayesianNetworks.factor_score) == [1, 8, 2, 3, 4, 5, 6, 7]

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
    @test EnhancedBayesianNetworks.sort_nodes(ig, ns, EnhancedBayesianNetworks.fill_factor_score) == [1, 8, 3, 7, 2, 4, 5, 6]
end