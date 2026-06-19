@testitem "Inference - apply_evidence" begin

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

    factors = EnhancedBayesianNetworks.factorize(bn)
    evidence_idx = [(4, 2)]
    EnhancedBayesianNetworks.apply_evidence!(factors, evidence_idx)
    @test factors[1].vars == [1]
    @test factors[2].vars == [1, 2]
    @test factors[3].vars == [1, 3]
    @test factors[4].vars == [3, 2]

    @test size(factors[4].table) == (3, 2)
    @test factors[4].table[1, 2] ≈ 0.8
end

@testitem "Inference - eliminate variable" begin

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

    factors = EnhancedBayesianNetworks.factorize(bn)
    newfactors = EnhancedBayesianNetworks.eliminate_var(factors, 10)
    @test newfactors === factors

    newfactors = EnhancedBayesianNetworks.eliminate_var(factors, 2)
    @test length(newfactors) == 3
    @test all(!EnhancedBayesianNetworks.containsvar(f, 2) for f in newfactors)
    @test sort.(getproperty.(newfactors, :vars)) == [[1], [1, 3], [1, 3, 4]]

    newfactors = EnhancedBayesianNetworks.eliminate_var(factors, 2)
    @test size(newfactors[3].table) == (2, 3, 2)
    expected = zeros(2, 3, 2)
    # W=Cloudy
    expected[1, 1, 1] = 0.36
    expected[1, 1, 2] = 0.64
    expected[1, 2, 1] = 0.52
    expected[1, 2, 2] = 0.48
    expected[1, 3, 1] = 0.516
    expected[1, 3, 2] = 0.484
    # W=Sunny
    expected[2, 1, 1] = 0.22
    expected[2, 1, 2] = 0.78
    expected[2, 2, 1] = 0.765
    expected[2, 2, 2] = 0.235
    expected[2, 3, 1] = 0.852
    expected[2, 3, 2] = 0.148
    @test newfactors[3].table ≈ expected
end
