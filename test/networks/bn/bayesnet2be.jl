@testset "Bayesian Networks 2 be" begin

    W = :Weather
    S = :Sprinkler
    R = :Rain
    G = :Grass
    L = :Rain

    nodes = [W, S, R, G, L]
    @test_throws ErrorException("network nodes names must be unique") BayesianNetwork2be(nodes)

    nodes = [W, S, R, G]
    bn = BayesianNetwork2be(nodes)
    add_child!(bn, :Weather, :Sprinkler)
    add_child!(bn, :Weather, :Rain)
    add_child!(bn, :Sprinkler, :Grass)
    add_child!(bn, :Rain, :Grass)
    order!(bn)
    @test bn.A == sparse([
        0.0 1.0 1.0 0.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0
    ])
    @test bn.topology == Dict(:Rain => 3, :Grass => 4, :Weather => 1, :Sprinkler => 2)
    @test issetequal(bn.nodes, nodes)

    @testset "add_child!" begin

        nodes = [W, S, R, G]
        net = BayesianNetwork2be(nodes)
        topology = Dict(:Weather => 1, :Sprinkler => 2, :Grass => 4, :Rain => 3)
        A = spzeros(4, 4)
        @test net.topology == topology
        @test issetequal(net.nodes, nodes)
        @test net.A == A

        @test_throws ErrorException("Recursion on the same node 'Weather' is not allowed in BayesianNetworks") add_child!(net, W, W)

        net_new1 = deepcopy(net)
        net_new2 = deepcopy(net)
        add_child!(net, W, R)
        A_net = sparse([
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0
        ])
        @test net.topology == topology
        @test net.nodes == nodes
        @test net.A == A_net
        add_child!(net_new1, W, R)
        @test net_new1 == net
        add_child!(net_new2, 1, 3)
        add_child!(net_new2, :Weather, :Rain)
        @test net_new2 == net
    end

    @testset "parents & children" begin

        nodes = [W, S, R, G]
        net = BayesianNetwork2be(nodes)
        add_child!(net, :Weather, :Rain)
        add_child!(net, :Weather, :Sprinkler)
        add_child!(net, :Sprinkler, :Grass)
        add_child!(net, :Rain, :Grass)

        @test parents(net, :Weather) == (Int64[], Symbol[])
        @test parents(net, 1) == (Int64[], Symbol[])
        @test parents(net, W) == (Int64[], Symbol[])
        @test parents(net, :Grass) == ([2, 3], [:Sprinkler, :Rain])
        @test parents(net, 4) == ([2, 3], [:Sprinkler, :Rain])
        @test parents(net, G) == ([2, 3], [:Sprinkler, :Rain])

        @test children(net, :Weather) == ([2, 3], [:Sprinkler, :Rain])
        @test children(net, 1) == ([2, 3], [:Sprinkler, :Rain])
        @test children(net, W) == ([2, 3], [:Sprinkler, :Rain])
        @test children(net, :Grass) == (Int64[], Symbol[])
        @test children(net, 4) == (Int64[], Symbol[])
        @test children(net, G) == (Int64[], Symbol[])
    end

    @testset "order network" begin

        root = :w
        child1 = :s
        child2 = :r
        grass = :g

        nodes = [root, child1, child2, grass]
        net = BayesianNetwork2be(nodes)
        add_child!(net, :w, :s)
        add_child!(net, :w, :r)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)
        order!(net)
        @test net.A == sparse(Matrix([0 1.0 1.0 0; 0 0 0 1.0; 0 0 0 1.0; 0 0 0 0]))
        @test net.topology == Dict(:w => 1, :s => 2, :g => 4, :r => 3)
        @test net.nodes == [root, child1, child2, grass]
    end
end

