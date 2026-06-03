@testitem "Bayesian Networks" begin
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

    nodes = [v, v, s]
    @test_throws ErrorException("Invalid BN: duplicate node names [:V]") BayesianNetwork(nodes)

    h = DiscreteNode(:H)
    h[:H=>:yesV] = 0.1
    h[:H=>:noH] = 0.9
    nodes = [v, h]
    @test_throws ErrorException("Invalid BN: duplicate node states [:yesV]") BayesianNetwork(nodes)

    nodes = [v, s, g]
    @test_throws ErrorException("Invalid BN: node/s [:G] are imprecise; CrealNetwork structure is required") BayesianNetwork(nodes)

    nodes = [v, s, t, l]
    bn = BayesianNetwork(nodes)
    @test isa(bn, BayesianNetwork)
    @test isa(bn, EnhancedBayesianNetworks.AbstractNetwork)
    @test issetequal(bn.nodes, nodes)
    @test bn.topology == Dict(:V => 1, :S => 2, :T => 3, :L => 4)
    @test bn.A == spzeros(Bool, 4, 4)

    @test_throws ErrorException("Invalid Network: node '[:V]' has a loop") add_child!(bn, v, v)

    @test_throws ErrorException("Invalid Network: node :T does not have the node :S in its CPT") add_child!(bn, s, t)

    @test_throws ErrorException("Nodes [:G] are not defined in the BN") add_child!(bn, v, g)

    @test_throws ErrorException("Nodes [:G] are not defined in the BN") add_child!(bn, :V, :G)

    add_child!(bn, v, t)
    @test bn.A == sparse([1], [3], [true], 4, 4)
    add_child!(bn, :S, :L)
    @test bn.A == sparse([1, 2], [3, 4], [true, true], 4, 4)


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
    sprinkler[:W=>:Cloudy, :S=>:Off] = 0.6
    sprinkler[:W=>:Sunny, :S=>:On] = 0.7
    sprinkler[:W=>:Sunny, :S=>:Off] = 0.3

    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:Yes, :S=>:On, :G=>:Wet] = 0.99
    grass[:R=>:Yes, :S=>:On, :G=>:Dry] = 0.01
    grass[:R=>:Yes, :S=>:Off, :G=>:Wet] = 0.9
    grass[:R=>:Yes, :S=>:Off, :G=>:Dry] = 0.1
    grass[:R=>:No, :S=>:On, :G=>:Wet] = 0.9
    grass[:R=>:No, :S=>:On, :G=>:Dry] = 0.1
    grass[:R=>:No, :S=>:Off, :G=>:Wet] = 0.1
    grass[:R=>:No, :S=>:Off, :G=>:Dry] = 0.9

    nodes = [weather, rain, sprinkler, grass]
    bn = BayesianNetwork(nodes)
    add_child!(bn, :W, :R)
    add_child!(bn, :W, :S)
    add_child!(bn, :R, :G)
    add_child!(bn, :S, :G)

    scenario1 = Evidence(:W => :Cloudy, :G => :Wet)
    @test_throws ErrorException("Node(s) [:R, :S] are not defined in the scenario Dict(:G => :Wet, :W => :Cloudy). Use Inference instead") joint_probability(bn, scenario1)

    scenario2 = Evidence(:W => :Cloudy, :G => :Wet, :R => :Yes, :S => :On, :N => :nothing)
    @test_logs (:warn, "Defined scenario contains Set([:N]) that are not defined in the BN. Therefore is useless for the scenario probability evaluation") joint_probability(bn, scenario2)

    scenario3 = Evidence(:W => :Cloudy, :G => :Mild, :R => :Yes, :S => :On)
    @test_throws ErrorException("Scenario defined state Mild for node G that does not belongs to its possible states [:Wet, :Dry]") joint_probability(bn, scenario3)

    scenario4 = Evidence(:W => :Cloudy, :G => :Wet, :R => :Yes, :S => :On)
    @test isapprox(joint_probability(bn, scenario4), 0.1584)

    scenario5 = Evidence(:W => :Cloudy, :G => :Dry, :R => :Yes, :S => :On)
    @test isapprox(joint_probability(bn, scenario5), 0.0016)
end

