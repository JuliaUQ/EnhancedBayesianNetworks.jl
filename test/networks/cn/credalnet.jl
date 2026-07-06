@testitem "Credal Network" setup=[ExtraDeps] begin
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

    h = DiscreteNode(:H)
    h[:H=>:yesH] = 0.01
    h[:H=>:noH] = 0.99

    nodes = [r, v, s, t]
    @test_throws MethodError CredalNetwork(nodes)

    nodes = [f1, v, s, t]
    @test_throws MethodError CredalNetwork(nodes)

    nodes = [v, s, t, l]
    @test_logs (:warn, "All the nodes are precise; BayesianNetwork structure should be used instead") CredalNetwork(nodes)

    nodes = [v, g, g]
    @test_throws ErrorException("Invalid CN: duplicate node names [:G]") CredalNetwork(nodes)

    i = DiscreteNode(:I)
    i[:I=>:g1] = Interval(0.2, 0.3)
    i[:I=>:i2] = Interval(0.7, 0.8)
    nodes = [v, g, i]
    @test_throws ErrorException("Invalid CN: duplicate node states [:g1]") CredalNetwork(nodes)

    nodes = [v, s, t, l, g]
    cn = CredalNetwork(nodes)
    @test isa(cn, CredalNetwork)
    @test isa(cn, EnhancedBayesianNetworks.AbstractNetwork)
    @test issetequal(cn.nodes, nodes)
    @test cn.topology == Dict(:V => 1, :S => 2, :T => 3, :L => 4, :G => 5)
    @test cn.A == spzeros(Bool, 5, 5)

    @test_throws ErrorException("Invalid Network: nodes [:V] have a loop") add_child!(cn, v, v)

    @test_throws ErrorException("Invalid Network: node :T does not have the node :S in its CPT") add_child!(cn, s, t)

    @test_throws ErrorException("Invalid Network: nodes [:H] are not defined in the network") add_child!(cn, v, h)

    @test_throws ErrorException("Invalid Network: nodes [:H] are not defined in the network") add_child!(cn, :V, :H)

    add_child!(cn, v, t)
    @test cn.A == sparse([1], [3], [true], 5, 5)
    add_child!(cn, :S, :L)
    @test cn.A == sparse([1, 2], [3, 4], [true, true], 5, 5)

    cn = @test_logs (:warn, "All the nodes are precise; BayesianNetwork structure should be used instead") CredalNetwork(DiscreteNode[])
    @test isempty(cn.nodes)
    @test size(cn.A) == (0, 0)
end