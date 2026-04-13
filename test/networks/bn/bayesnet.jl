@testset "Bayesian Networks" begin
    r = ContinuousNode(:R, Normal())

    v = DiscreteNode(:V)
    v[:V=>:yesV] = 0.01
    v[:V=>:noV] = 0.99

    s = DiscreteNode(:S)
    s[:S=>:yesS] = 0.5
    s[:S=>:noS] = 0.5

    t = DiscreteNode(:T, [:V])
    t[:V=>:yesV, :T=>:yesT] = 0.05
    t[:V=>:yesV, :T=>:noT] = 0.95
    t[:V=>:noV, :T=>:yesT] = 0.01
    t[:V=>:noV, :T=>:noT] = 0.99

    l = DiscreteNode(:L, [:S])
    l[:S=>:yesS, :L=>:yesL] = 0.1
    l[:S=>:yesS, :L=>:noL] = 0.9
    l[:S=>:noS, :L=>:yesL] = 0.01
    l[:S=>:noS, :L=>:noL] = 0.99

    f1 = DiscreteFunctionalNode(
        :F1, [Model(df -> df.v1 .+ df.R, :f1)], df -> 0.8 .- df.f1, MonteCarlo(200)
    )

    g = DiscreteNode(:G)
    g[:G=>:g1] = Interval(0.1, 0.2)
    g[:G=>:g2] = Interval(0.15, 0.25)
    g[:G=>:g3] = 0.2
    g[:G=>:g4] = Interval(0.3, 0.5)

    nodes = [r, v, s, t]
    @test_throws MethodError BayesianNetwork(nodes)

    nodes = [f1, v, s, t]
    @test_throws MethodError BayesianNetwork(nodes)

    nodes = [v, s, g]
    @test_throws ErrorException("Invalid BN: node/s [:G] are imprecise; CrealNetwork structure is required") BayesianNetwork(nodes)

    nodes = [v, s, t, l]
    bn = BayesianNetwork(nodes)
    @test isa(bn, BayesianNetwork)
    @test isa(bn, EnhancedBayesianNetworks.AbstractNetwork)
    @test issetequal(bn.nodes, nodes)
    @test bn.topology == Dict(:V => 1, :S => 2, :T => 3, :L => 4)
    @test bn.A == spzeros(Bool, 4, 4)

    @test_throws ErrorException("Invalid eBN: node '[:V]' have recursion") add_child!(bn, v, v)

    @test_throws ErrorException("Invalid Network: node T does not have the node(s) S in its CPT") add_child!(bn, s, t)


    @test_throws ErrorException("node(s) [:G] is (are) not defined in the BN") add_child!(bn, v, g)

    @test_throws ErrorException("node(s) [:G] is (are) not defined in the BN") add_child!(bn, :V, :G)

    add_child!(bn, v, t)
    @test bn.A == sparse([1], [3], [true], 4, 4)
    add_child!(bn, :S, :L)
    @test bn.A == sparse([1, 2], [3, 4], [true, true], 4, 4)
end

