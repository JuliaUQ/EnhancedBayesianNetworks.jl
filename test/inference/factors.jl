@testitem "Inference - Factor & factorize" setup=[ExtraDeps] begin
    f = Factor(
        [1],
        [0.5, 0.5]
    )
    @test isa(f, Factor)
    @test f.vars == [1]
    @test f.table == [0.5, 0.5]

    f = Factor(
        [1, 2],
        reshape(collect(1:4), 2, 2)
    )
    @test isa(f, Factor)
    @test f.vars == [1, 2]
    @test f.table == reshape(collect(1:4), 2, 2)

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
    ns = EnhancedBayesianNetworks.NetworkSchema(bn)

    f1 = EnhancedBayesianNetworks.factorize(weather, ns)
    @test f1.vars == [1]
    @test f1.table == [0.5, 0.5]

    f2 = EnhancedBayesianNetworks.factorize(rain, ns)
    @test f2.vars == [1, 2]
    @test f2.table == [0.8 0.2; 0.1 0.9]

    factors = EnhancedBayesianNetworks.factorize(bn)
    @test factors[1].vars == [1]
    @test factors[1].table == [0.5, 0.5]
    @test factors[2].vars == [1, 2]
    @test factors[2].table == [0.8 0.2; 0.1 0.9]
    @test factors[3].vars == [1, 3]
    @test factors[3].table == [0.4 0.4 0.2; 0.6000000000000001 0.30000000000000004 0.10000000000000002]
    @test factors[4].vars == [3, 2, 4]
    @test factors[4].table == [0.4 0.2; 0.45 0.8; 0.42 0.9;;; 0.6 0.8; 0.55 0.2; 0.58 0.1]

    @test EnhancedBayesianNetworks.varpos(factors[1], 1) == 1
    @test EnhancedBayesianNetworks.varpos(factors[4], 2) == 2
    @test isnothing(EnhancedBayesianNetworks.varpos(factors[1], 2))
    @test EnhancedBayesianNetworks.containsvar(factors[1], 1)
    @test !EnhancedBayesianNetworks.containsvar(factors[1], 2)
end