@testsnippet SetupCommonNetTest begin

    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = DoubleLoop(MonteCarlo(100))
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)

    grass_incomplete = DiscreteNode(:G, [:S, :R])
    grass_incomplete[:R=>:yes, :S=>:on, :G=>:dry] = 0
    grass_incomplete[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_incomplete[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_incomplete[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_incomplete[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_incomplete[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_incomplete[:R=>:no, :S=>:off, :G=>:wet] = 0

    grass_not_exhaustive = DiscreteNode(:G, [:S, :R])
    grass_not_exhaustive[:R=>:yes, :S=>:on, :G=>:dry] = 0
    grass_not_exhaustive[:R=>:yes, :S=>:on, :G=>:wet] = 0.999
    grass_not_exhaustive[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_not_exhaustive[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_not_exhaustive[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_not_exhaustive[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_not_exhaustive[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_not_exhaustive[:R=>:no, :S=>:off, :G=>:wet] = 0

    grass_not_mutually_exclusive = DiscreteNode(:G, [:S, :R])
    grass_not_mutually_exclusive[:R=>:yes, :S=>:on, :G=>:dry] = 0.3
    grass_not_mutually_exclusive[:R=>:yes, :S=>:on, :G=>:wet] = 0.999
    grass_not_mutually_exclusive[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_not_mutually_exclusive[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_not_mutually_exclusive[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_not_mutually_exclusive[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_not_mutually_exclusive[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_not_mutually_exclusive[:R=>:no, :S=>:off, :G=>:wet] = 0

    grass_ub = DiscreteNode(:G, [:S, :R])
    grass_ub[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
    grass_ub[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.2, 0.3)
    grass_ub[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_ub[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_ub[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_ub[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_ub[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_ub[:R=>:no, :S=>:off, :G=>:wet] = 0

    grass_ub2 = DiscreteNode(:G, [:S, :R])
    grass_ub2[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
    grass_ub2[:R=>:yes, :S=>:on, :G=>:wet] = 0.2
    grass_ub2[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_ub2[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_ub2[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_ub2[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_ub2[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_ub2[:R=>:no, :S=>:off, :G=>:wet] = 0

    grass_lb = DiscreteNode(:G, [:S, :R])
    grass_lb[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
    grass_lb[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.8, 0.9)
    grass_lb[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_lb[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_lb[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_lb[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_lb[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_lb[:R=>:no, :S=>:off, :G=>:wet] = 0

    grass_lb2 = DiscreteNode(:G, [:S, :R])
    grass_lb2[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
    grass_lb2[:R=>:yes, :S=>:on, :G=>:wet] = 0.8
    grass_lb2[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass_lb2[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass_lb2[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass_lb2[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass_lb2[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass_lb2[:R=>:no, :S=>:off, :G=>:wet] = 0

end

@testitem "Networks Common - cyclicality & connection" setup=[SetupSprinklereBN, SetupCommonNetTest] begin
    ## BN
    A = DiscreteNode(:A, [:B])
    A[:B=>:b1, :A=>:a1] = 0.05
    A[:B=>:b1, :A=>:a2] = 0.95
    A[:B=>:b2, :A=>:a1] = 0.7
    A[:B=>:b2, :A=>:a2] = 0.3
    B = DiscreteNode(:B, [:A])
    B[:B=>:b1, :A=>:a1] = 0.05
    B[:B=>:b1, :A=>:a2] = 0.95
    B[:B=>:b2, :A=>:a1] = 0.7
    B[:B=>:b2, :A=>:a2] = 0.3
    net = BayesianNetwork([A, B, weather])
    add_child!(net, A, B)
    add_child!(net, B, A)
    @test EnhancedBayesianNetworks.iscyclic(net)
    @test !EnhancedBayesianNetworks.isconnected(net)

    ## eBN
    net = EnhancedBayesianNetwork([A, B, weather])
    add_child!(net, A, B)
    add_child!(net, B, A)
    @test EnhancedBayesianNetworks.iscyclic(net)
    @test !EnhancedBayesianNetworks.isconnected(net)

    ## CN
    B = DiscreteNode(:B, [:A])
    B[:B=>:b1, :A=>:a1] = Interval(0.05, 0.1)
    B[:B=>:b1, :A=>:a2] = Interval(0.6, 0.95)
    B[:B=>:b2, :A=>:a1] = 0.7
    B[:B=>:b2, :A=>:a2] = 0.3
    net = CredalNetwork([A, B, weather])
    add_child!(net, A, B)
    add_child!(net, B, A)
    @test EnhancedBayesianNetworks.iscyclic(net)
    @test !EnhancedBayesianNetworks.isconnected(net)
end

@testitem "Networks Common - parents, children and ancestors" setup=[SetupSprinklereBN, SetupCommonNetTest] begin
    ## BN
    nodes = [weather, grass, rain, sprinkler]
    net = BayesianNetwork(nodes)
    add_child!(net, weather, [rain, sprinkler])
    add_child!(net, [rain, sprinkler], grass)
    @test !EnhancedBayesianNetworks.iscyclic(net)
    @test EnhancedBayesianNetworks.isconnected(net)
    @test isempty(parents(net, :W))
    @test issetequal(parents(net, :G), [:R, :S])
    @test issetequal(parents(net, grass), [:R, :S])
    @test isempty(children(net, :G))
    @test issetequal(children(net, :W), [:R, :S])
    @test issetequal(children(net, weather), [:R, :S])

    ## CN
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = Interval(0.4, 0.6)
    weather[:W=>:cloudy] = Interval(0.4, 0.6)

    nodes = [weather, grass, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [rain, sprinkler])
    add_child!(net, [rain, sprinkler], grass)
    @test !EnhancedBayesianNetworks.iscyclic(net)
    @test EnhancedBayesianNetworks.isconnected(net)
    @test isempty(parents(net, :W))
    @test issetequal(parents(net, :G), [:R, :S])
    @test issetequal(parents(net, grass), [:R, :S])
    @test isempty(children(net, :G))
    @test issetequal(children(net, :W), [:R, :S])
    @test issetequal(children(net, weather), [:R, :S])

    ## eBN
    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [rain, sprinkler])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    @test !EnhancedBayesianNetworks.iscyclic(net)
    @test EnhancedBayesianNetworks.isconnected(net)
    @test isempty(parents(net, :W))
    @test issetequal(parents(net, :G), [:R, :S])
    @test issetequal(parents(net, grass), [:R, :S])
    @test isempty(children(net, :G))
    @test issetequal(children(net, :W), [:R, :S])
    @test issetequal(children(net, weather), [:R, :S])
    @test issetequal(discrete_ancestors(net, grass), [:R, :S])
    @test issetequal(discrete_ancestors(net, :G), [:R, :S])

    rain2 = ContinuousNode(:Rc, [:W])
    rain2[:W=>:sunny] = Normal()
    rain2[:W=>:cloudy] = Normal()
    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain, rain2])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    @test issetequal(discrete_ancestors(net, grass2), [:W, :S])
    @test issetequal(discrete_ancestors(net, :G2), [:W, :S])
end

@testitem "Networks Common - verify BN" setup=[SetupSprinklereBN, SetupCommonNetTest] begin
    nodes = [weather, grass, rain, sprinkler]
    net = BayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, sprinkler, grass)
    @test_throws ErrorException("Invalid CPT: node :G has nodes [:R] defined in the CPT only, but they have not been added via add_child!") EnhancedBayesianNetworks.verify_parents(net, grass)
    add_child!(net, rain, grass)
    @test isnothing(EnhancedBayesianNetworks.verify_parents(net, grass))

    nodes = [weather, grass_incomplete, rain, sprinkler]
    net = BayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_incomplete)
    @test_throws ErrorException("Invalid CPT: node :G is missing the following scenario [:R => :yes, :S => :on, :G => :wet]") EnhancedBayesianNetworks.verify_scenarios(net, grass_incomplete)
    nodes = [weather, grass, rain, sprinkler]
    net = BayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    @test isnothing(EnhancedBayesianNetworks.verify_scenarios(net, grass))

    nodes = [weather, grass_not_exhaustive, rain, sprinkler]
    net = BayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_not_exhaustive)
    @test_logs (:warn, "Node :G has CPT values [0, 0.999] for the scenario [:R => :yes, :S => :on] and will be normalized!") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_not_exhaustive)
    @test filter(grass_not_exhaustive.cpt, ([:S, :R, :G] .=> [:on, :yes, :wet])...).Π == [1.0]
    nodes = [weather, grass_not_mutually_exclusive, rain, sprinkler]
    net = BayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_not_mutually_exclusive)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [0.3, 0.999] not exhaustive and mutually exclusive for the scenario [:R => :yes, :S => :on]") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_not_mutually_exclusive)
end

@testitem "Networks Common - verify CN" setup=[SetupSprinklereBN, SetupCommonNetTest] begin
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = Interval(0.4, 0.6)
    weather[:W=>:cloudy] = Interval(0.4, 0.6)
    nodes = [weather, grass, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, sprinkler, grass)
    @test_throws ErrorException("Invalid CPT: node :G has nodes [:R] defined in the CPT only, but they have not been added via add_child!") EnhancedBayesianNetworks.verify_parents(net, grass)
    add_child!(net, rain, grass)
    @test isnothing(EnhancedBayesianNetworks.verify_parents(net, grass))

    nodes = [weather, grass_incomplete, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_incomplete)
    @test_throws ErrorException("Invalid CPT: node :G is missing the following scenario [:R => :yes, :S => :on, :G => :wet]") EnhancedBayesianNetworks.verify_scenarios(net, grass_incomplete)
    nodes = [weather, grass, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    @test isnothing(EnhancedBayesianNetworks.verify_scenarios(net, grass))

    nodes = [weather, grass_not_exhaustive, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_not_exhaustive)
    @test_logs (:warn, "Node :G has CPT values [0, 0.999] for the scenario [:R => :yes, :S => :on] and will be normalized!") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_not_exhaustive)
    @test filter(grass_not_exhaustive.cpt, ([:S, :R, :G] .=> [:on, :yes, :wet])...).Π == [1.0]
    nodes = [weather, grass_not_mutually_exclusive, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_not_mutually_exclusive)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [0.3, 0.999] not exhaustive and mutually exclusive for the scenario [:R => :yes, :S => :on]") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_not_mutually_exclusive)

    nodes = [weather, grass_ub, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_ub)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], [0.2, 0.3]] for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_ub)

    nodes = [weather, grass_ub2, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_ub2)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], 0.2] for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_ub2)

    nodes = [weather, grass_lb, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_lb)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], [0.8, 0.9]] for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_lb)

    nodes = [weather, grass_lb2, rain, sprinkler]
    net = CredalNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_lb2)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], 0.8] for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_lb2)
end

@testitem "Networks Common - verify eBN" setup=[SetupSprinklereBN, SetupCommonNetTest] begin
    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [sprinkler, rain2], grass2)
    add_child!(net, sprinkler, grass)
    @test_throws ErrorException("Invalid CPT: node :G has nodes [:R] defined in the CPT only, but they have not been added via add_child!") EnhancedBayesianNetworks.verify_parents(net, grass)
    add_child!(net, rain, grass)
    @test isnothing(EnhancedBayesianNetworks.verify_parents(net, grass))
    @test isnothing(EnhancedBayesianNetworks.verify_parents(net, grass2))

    nodes = [weather, grass_incomplete, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_incomplete)
    @test_throws ErrorException("Invalid CPT: node :G is missing the following scenario [:R => :yes, :S => :on, :G => :wet]") EnhancedBayesianNetworks.verify_scenarios(net, grass_incomplete)
    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [sprinkler, rain2], grass2)
    add_child!(net, [rain, sprinkler], grass)
    @test isnothing(EnhancedBayesianNetworks.verify_scenarios(net, grass))

    nodes = [weather, grass_not_exhaustive, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_not_exhaustive)
    @test_logs (:warn, "Node :G has CPT values [0, 0.999] for the scenario [:R => :yes, :S => :on] and will be normalized!") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_not_exhaustive)
    @test filter(grass_not_exhaustive.cpt, ([:S, :R, :G] .=> [:on, :yes, :wet])...).Π == [1.0]
    nodes = [weather, grass_not_mutually_exclusive, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_not_mutually_exclusive)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [0.3, 0.999] not exhaustive and mutually exclusive for the scenario [:R => :yes, :S => :on]") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_not_mutually_exclusive)

    nodes = [weather, grass_ub, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_ub)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], [0.2, 0.3]] for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_ub)

    nodes = [weather, grass_ub2, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_ub2)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], 0.2] for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_ub2)

    nodes = [weather, grass_lb, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_lb)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], [0.8, 0.9]] for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_lb)

    nodes = [weather, grass_lb2, rain, sprinkler]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass_lb2)
    @test_throws ErrorException("Invalid CPT: node :G has CPT values [[0.3, 0.4], 0.8] for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass_lb2)
end


@testitem "Networks Common - Markov Blanket" begin
    x1 = DiscreteNode(:x1)
    x1[:x1=>:x1y] = 0.5
    x1[:x1=>:x1n] = 0.5

    x2 = DiscreteNode(:x2)
    x2[:x2=>:x2y] = 0.5
    x2[:x2=>:x2n] = 0.5

    x4 = DiscreteNode(:x4)
    x4[:x4=>:x4y] = 0.5
    x4[:x4=>:x4n] = 0.5

    x8 = DiscreteNode(:x8)
    x8[:x8=>:x8y] = 0.5
    x8[:x8=>:x8n] = 0.5

    x3 = DiscreteNode(:x3, [:x1])
    x3[:x1=>:x1y, :x3=>:x3y] = 0.5
    x3[:x1=>:x1y, :x3=>:x3n] = 0.5
    x3[:x1=>:x1n, :x3=>:x3y] = 0.5
    x3[:x1=>:x1n, :x3=>:x3n] = 0.5

    x5 = DiscreteNode(:x5, [:x2])
    x5[:x2=>:x2y, :x5=>:x5y] = 0.5
    x5[:x2=>:x2y, :x5=>:x5n] = 0.5
    x5[:x2=>:x2n, :x5=>:x5y] = 0.5
    x5[:x2=>:x2n, :x5=>:x5n] = 0.5

    x7 = DiscreteNode(:x7, [:x4])
    x7[:x4=>:x4y, :x7=>:x7y] = 0.5
    x7[:x4=>:x4y, :x7=>:x7n] = 0.5
    x7[:x4=>:x4n, :x7=>:x7y] = 0.5
    x7[:x4=>:x4n, :x7=>:x7n] = 0.5

    x11 = DiscreteNode(:x11, [:x8])
    x11[:x8=>:x8y, :x11=>:x11y] = 0.5
    x11[:x8=>:x8y, :x11=>:x11n] = 0.5
    x11[:x8=>:x8n, :x11=>:x11y] = 0.5
    x11[:x8=>:x8n, :x11=>:x11n] = 0.5

    x6 = DiscreteNode(:x6, [:x3, :x4])
    x6[:x3=>:x3y, :x4=>:x4y, :x6=>:x6y] = 0.5
    x6[:x3=>:x3y, :x4=>:x4y, :x6=>:x6n] = 0.5
    x6[:x3=>:x3y, :x4=>:x4n, :x6=>:x6y] = 0.5
    x6[:x3=>:x3y, :x4=>:x4n, :x6=>:x6n] = 0.5
    x6[:x3=>:x3n, :x4=>:x4y, :x6=>:x6y] = 0.5
    x6[:x3=>:x3n, :x4=>:x4y, :x6=>:x6n] = 0.5
    x6[:x3=>:x3n, :x4=>:x4n, :x6=>:x6y] = 0.5
    x6[:x3=>:x3n, :x4=>:x4n, :x6=>:x6n] = 0.5

    x9 = DiscreteNode(:x9, [:x5, :x6])
    x9[:x5=>:x5y, :x6=>:x6y, :x9=>:x9y] = 0.5
    x9[:x5=>:x5y, :x6=>:x6y, :x9=>:x9n] = 0.5
    x9[:x5=>:x5y, :x6=>:x6n, :x9=>:x9y] = 0.5
    x9[:x5=>:x5y, :x6=>:x6n, :x9=>:x9n] = 0.5
    x9[:x5=>:x5n, :x6=>:x6y, :x9=>:x9y] = 0.5
    x9[:x5=>:x5n, :x6=>:x6y, :x9=>:x9n] = 0.5
    x9[:x5=>:x5n, :x6=>:x6n, :x9=>:x9y] = 0.5
    x9[:x5=>:x5n, :x6=>:x6n, :x9=>:x9n] = 0.5

    x10 = DiscreteNode(:x10, [:x8, :x6])
    x10[:x8=>:x8y, :x6=>:x6y, :x10=>:x10y] = 0.5
    x10[:x8=>:x8y, :x6=>:x6y, :x10=>:x10n] = 0.5
    x10[:x8=>:x8y, :x6=>:x6n, :x10=>:x10y] = 0.5
    x10[:x8=>:x8y, :x6=>:x6n, :x10=>:x10n] = 0.5
    x10[:x8=>:x8n, :x6=>:x6y, :x10=>:x10y] = 0.5
    x10[:x8=>:x8n, :x6=>:x6y, :x10=>:x10n] = 0.5
    x10[:x8=>:x8n, :x6=>:x6n, :x10=>:x10y] = 0.5
    x10[:x8=>:x8n, :x6=>:x6n, :x10=>:x10n] = 0.5

    x12 = DiscreteNode(:x12, [:x9])
    x12[:x9=>:x9y, :x12=>:x12y] = 0.5
    x12[:x9=>:x9y, :x12=>:x12n] = 0.5
    x12[:x9=>:x9n, :x12=>:x12y] = 0.5
    x12[:x9=>:x9n, :x12=>:x12n] = 0.5

    x13 = DiscreteNode(:x13, [:x10])
    x13[:x10=>:x10y, :x13=>:x13y] = 0.5
    x13[:x10=>:x10y, :x13=>:x13n] = 0.5
    x13[:x10=>:x10n, :x13=>:x13y] = 0.5
    x13[:x10=>:x10n, :x13=>:x13n] = 0.5

    ## BN
    nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]
    net = BayesianNetwork(nodes)
    add_child!(net, :x1, :x3)
    add_child!(net, :x2, :x5)
    add_child!(net, :x4, :x7)
    add_child!(net, :x8, :x11)
    add_child!(net, :x3, :x6)
    add_child!(net, :x4, :x6)
    add_child!(net, :x5, :x9)
    add_child!(net, :x6, :x9)
    add_child!(net, :x6, :x10)
    add_child!(net, :x8, :x10)
    add_child!(net, :x9, :x12)
    add_child!(net, :x10, :x13)

    @test issetequal(markov_blanket(net, :x6), [:x3, :x4, :x5, :x8, :x9, :x10])
    @test issetequal(markov_blanket(net, x6), [:x3, :x4, :x5, :x8, :x9, :x10])

    ## CN
    x1 = DiscreteNode(:x1)
    x1[:x1=>:x1y] = Interval(0.4, 0.6)
    x1[:x1=>:x1n] = Interval(0.4, 0.6)
    nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]
    net = CredalNetwork(nodes)
    add_child!(net, :x1, :x3)
    add_child!(net, :x2, :x5)
    add_child!(net, :x4, :x7)
    add_child!(net, :x8, :x11)
    add_child!(net, :x3, :x6)
    add_child!(net, :x4, :x6)
    add_child!(net, :x5, :x9)
    add_child!(net, :x6, :x9)
    add_child!(net, :x6, :x10)
    add_child!(net, :x8, :x10)
    add_child!(net, :x9, :x12)
    add_child!(net, :x10, :x13)

    @test issetequal(markov_blanket(net, :x6), [:x3, :x4, :x5, :x8, :x9, :x10])
    @test issetequal(markov_blanket(net, x6), [:x3, :x4, :x5, :x8, :x9, :x10])

    ## eBN
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, :x1, :x3)
    add_child!(net, :x2, :x5)
    add_child!(net, :x4, :x7)
    add_child!(net, :x8, :x11)
    add_child!(net, :x3, :x6)
    add_child!(net, :x4, :x6)
    add_child!(net, :x5, :x9)
    add_child!(net, :x6, :x9)
    add_child!(net, :x6, :x10)
    add_child!(net, :x8, :x10)
    add_child!(net, :x9, :x12)
    add_child!(net, :x10, :x13)

    @test issetequal(markov_blanket(net, :x6), [:x3, :x4, :x5, :x8, :x9, :x10])
    @test issetequal(markov_blanket(net, x6), [:x3, :x4, :x5, :x8, :x9, :x10])
end

@testitem "Networks Common - add/remove node" setup=[ExtraDeps, SetupSprinklereBN, SetupCommonNetTest] begin
    ## BN
    nodes = [weather, sprinkler, rain, grass]
    net = BayesianNetwork(nodes)
    add_child!(net, :W, :S)
    add_child!(net, :W, :R)
    add_child!(net, :S, :G)
    add_child!(net, :R, :G)
    net1 = deepcopy(net)
    net2 = deepcopy(net)
    net3 = deepcopy(net)

    EnhancedBayesianNetworks.remove_node!(net1, grass)
    EnhancedBayesianNetworks.remove_node!(net2, :G)

    @test issetequal(getproperty.(net1.nodes, :name), getproperty.([weather, sprinkler, rain], :name))
    adj = sparse([1, 1], [2, 3], [true, true], 3, 3)
    @test net1.A == adj
    @test net1.topology == Dict(:W => 1, :R => 3, :S => 2)
    @test net2.A == net1.A
    @test issetequal(getproperty.(net2.nodes, :name), getproperty.([weather, sprinkler, rain], :name))
    @test net2.topology == net1.topology
    @test isnothing(EnhancedBayesianNetworks.remove_node!(net3, :G))

    net4 = deepcopy(net1)
    net5 = deepcopy(net1)

    EnhancedBayesianNetworks.add_node!(net4, grass)
    @test issetequal(getproperty.(net4.nodes, :name), [:W, :R, :S, :G])
    adj = sparse([1, 1], [2, 3], [true, true], 4, 4)
    @test net4.A == adj
    @test net4.topology == Dict(:W => 1, :R => 3, :S => 2, :G => 4)

    ## CN
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = Interval(0.4, 0.6)
    weather[:W=>:cloudy] = Interval(0.4, 0.6)
    nodes = [weather, sprinkler, rain, grass]
    net = CredalNetwork(nodes)
    add_child!(net, :W, :S)
    add_child!(net, :W, :R)
    add_child!(net, :S, :G)
    add_child!(net, :R, :G)
    net1 = deepcopy(net)
    net2 = deepcopy(net)
    net3 = deepcopy(net)

    EnhancedBayesianNetworks.remove_node!(net1, grass)
    EnhancedBayesianNetworks.remove_node!(net2, :G)

    @test issetequal(getproperty.(net1.nodes, :name), getproperty.([weather, sprinkler, rain], :name))
    adj = sparse([1, 1], [2, 3], [true, true], 3, 3)
    @test net1.A == adj
    @test net1.topology == Dict(:W => 1, :R => 3, :S => 2)
    @test net2.A == net1.A
    @test issetequal(getproperty.(net2.nodes, :name), getproperty.([weather, sprinkler, rain], :name))
    @test net2.topology == net1.topology
    @test isnothing(EnhancedBayesianNetworks.remove_node!(net3, :G))

    net4 = deepcopy(net1)
    net5 = deepcopy(net1)

    EnhancedBayesianNetworks.add_node!(net4, grass)
    @test issetequal(getproperty.(net4.nodes, :name), [:W, :R, :S, :G])
    adj = sparse([1, 1], [2, 3], [true, true], 4, 4)
    @test net4.A == adj
    @test net4.topology == Dict(:W => 1, :R => 3, :S => 2, :G => 4)

    ## eBN
    nodes = [weather, sprinkler, rain, grass]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, :W, :S)
    add_child!(net, :W, :R)
    add_child!(net, :S, :G)
    add_child!(net, :R, :G)
    net1 = deepcopy(net)
    net2 = deepcopy(net)
    net3 = deepcopy(net)

    EnhancedBayesianNetworks.remove_node!(net1, grass)
    EnhancedBayesianNetworks.remove_node!(net2, :G)

    @test issetequal(getproperty.(net1.nodes, :name), getproperty.([weather, sprinkler, rain], :name))
    adj = sparse([1, 1], [2, 3], [true, true], 3, 3)
    @test net1.A == adj
    @test net1.topology == Dict(:W => 1, :R => 3, :S => 2)
    @test net2.A == net1.A
    @test issetequal(getproperty.(net2.nodes, :name), getproperty.([weather, sprinkler, rain], :name))
    @test net2.topology == net1.topology
    @test isnothing(EnhancedBayesianNetworks.remove_node!(net3, :G))

    net4 = deepcopy(net1)
    net5 = deepcopy(net1)

    EnhancedBayesianNetworks.add_node!(net4, grass)
    @test issetequal(getproperty.(net4.nodes, :name), [:W, :R, :S, :G])
    adj = sparse([1, 1], [2, 3], [true, true], 4, 4)
    @test net4.A == adj
    @test net4.topology == Dict(:W => 1, :R => 3, :S => 2, :G => 4)
end

@testitem "Networks Common - sorting functions" setup=[ExtraDeps] begin
    A = DiscreteNode(:A, [:B])
    A[:B=>:b1, :A=>:a1] = 0.05
    A[:B=>:b1, :A=>:a2] = 0.95
    A[:B=>:b2, :A=>:a1] = 0.7
    A[:B=>:b2, :A=>:a2] = 0.3
    B = DiscreteNode(:B, [:A])
    B[:B=>:b1, :A=>:a1] = 0.05
    B[:B=>:b1, :A=>:a2] = 0.95
    B[:B=>:b2, :A=>:a1] = 0.7
    B[:B=>:b2, :A=>:a2] = 0.3

    C = DiscreteNode(:C)
    C[:C=>:C1] = 0.1
    C[:C=>:C2] = 0.9
    D = DiscreteNode(:D)
    D[:D=>:D1] = 0.1
    D[:D=>:D2] = 0.9
    E = DiscreteNode(:E, [:D])
    E[:E=>:E1, :D=>:D1] = 0.05
    E[:E=>:E1, :D=>:D2] = 0.95
    E[:E=>:E2, :D=>:D1] = 0.7
    E[:E=>:E2, :D=>:D2] = 0.3

    net = BayesianNetwork([A, B])
    add_child!(net, A, B)
    add_child!(net, B, A)
    @test_throws ErrorException("Invalid Network: network is cyclic!") order!(net)
    net = BayesianNetwork([C, D, E])
    add_child!(net, D, E)
    @test_throws ErrorException("Invalid Network: network is not connected") order!(net)

    ## CN
    A = DiscreteNode(:A, [:B])
    A[:B=>:b1, :A=>:a1] = Interval(0.05, 0.1)
    A[:B=>:b1, :A=>:a2] = Interval(0.6, 0.95)
    A[:B=>:b2, :A=>:a1] = 0.7
    A[:B=>:b2, :A=>:a2] = 0.3

    C = DiscreteNode(:C)
    C[:C=>:C1] = Interval(0.4, 0.6)
    C[:C=>:C2] = Interval(0.4, 0.6)

    net = CredalNetwork([A, B])
    add_child!(net, A, B)
    add_child!(net, B, A)
    @test_throws ErrorException("Invalid Network: network is cyclic!") order!(net)
    net = CredalNetwork([C, D, E])
    add_child!(net, D, E)
    @test_throws ErrorException("Invalid Network: network is not connected") order!(net)

    ## eBN
    net = EnhancedBayesianNetwork([A, B])
    add_child!(net, A, B)
    add_child!(net, B, A)
    @test_throws ErrorException("Invalid Network: network is cyclic!") order!(net)
    net = EnhancedBayesianNetwork([C, D, E])
    add_child!(net, D, E)
    @test_throws ErrorException("Invalid Network: network is not connected") order!(net)

    A = DiscreteNode(:A)
    A[:A=>:a1] = 0.3
    A[:A=>:a2] = 0.7
    B = DiscreteNode(:B)
    B[:B=>:b1] = 0.3
    B[:B=>:b2] = 0.7
    C = ContinuousNode(:C)
    C[] = Normal()
    D = DiscreteNode(:D, [:A, :B])
    D[:A=>:a1, :B=>:b1, :D=>:d1] = 0.2
    D[:A=>:a1, :B=>:b1, :D=>:d2] = 0.8
    D[:A=>:a1, :B=>:b2, :D=>:d1] = 0.2
    D[:A=>:a1, :B=>:b2, :D=>:d2] = 0.8
    D[:A=>:a2, :B=>:b1, :D=>:d1] = 0.2
    D[:A=>:a2, :B=>:b1, :D=>:d2] = 0.8
    D[:A=>:a2, :B=>:b2, :D=>:d1] = 0.2
    D[:A=>:a2, :B=>:b2, :D=>:d2] = 0.8
    model = Model(df -> df.C .+ df.D, :E)
    performance = df -> df.E
    sim = MonteCarlo(100)
    E = DiscreteFunctionalNode(:E, model, performance, sim)
    net = EnhancedBayesianNetwork([E, A, C, D, B])
    add_child!(net, [A, B], D)
    add_child!(net, [D, C], E)

    EnhancedBayesianNetworks.topologically_sort!(net)
    @test getproperty.(net.nodes, :name) == [:A, :C, :B, :D, :E]
    @test net.topology == Dict(:A => 1, :B => 3, :C => 2, :D => 4, :E => 5)
    adj = spzeros(Bool, 5, 5)
    adj[1, 4] = true
    adj[2, 5] = true
    adj[3, 4] = true
    adj[4, 5] = true
    @test net.A == adj

    net = EnhancedBayesianNetwork([E, A, C, D, B])
    add_child!(net, [A, B], D)
    add_child!(net, [D, C], E)
    order!(net)
    @test getproperty.(net.nodes, :name) == [:A, :C, :B, :D, :E]
    @test net.topology == Dict(:A => 1, :B => 3, :C => 2, :D => 4, :E => 5)
    adj = spzeros(Bool, 5, 5)
    adj[1, 4] = true
    adj[2, 5] = true
    adj[3, 4] = true
    adj[4, 5] = true
    @test net.A == adj

    net = EnhancedBayesianNetwork([C, D, B, A])
    add_child!(net, [A, B], D)
    @test_throws ErrorException("Invalid Network: network is not connected") order!(net)

    net = EnhancedBayesianNetwork([E, A, C, D, B])
    add_child!(net, B, D)
    add_child!(net, [A, D, C], E)
    @test_throws ErrorException("Invalid CPT: node :D has nodes [:A] defined in the CPT only, but they have not been added via add_child!") order!(net)

    D[:A=>:a2, :B=>:b1, :D=>:d1] = 0.1
    net = EnhancedBayesianNetwork([E, A, C, D, B])
    add_child!(net, [A, B], D)
    add_child!(net, [D, C], E)
    @test_throws ErrorException("Invalid CPT: node :D has CPT values [0.1, 0.8] not exhaustive and mutually exclusive for the scenario [:A => :a2, :B => :b1]") order!(net)

    D = DiscreteNode(:D, [:A, :B])
    D[:A=>:a1, :B=>:b1, :D=>:d1] = 0.2
    D[:A=>:a1, :B=>:b1, :D=>:d2] = 0.8
    D[:A=>:a1, :B=>:b2, :D=>:d1] = 0.2
    D[:A=>:a2, :B=>:b1, :D=>:d1] = 0.2
    D[:A=>:a2, :B=>:b1, :D=>:d2] = 0.8
    D[:A=>:a2, :B=>:b2, :D=>:d1] = 0.2
    D[:A=>:a2, :B=>:b2, :D=>:d2] = 0.8
    net = EnhancedBayesianNetwork([E, A, C, D, B])
    add_child!(net, [A, B], D)
    add_child!(net, [D, C], E)
    @test_throws ErrorException("Invalid CPT: node :D is missing the following scenario [:A => :a1, :B => :b2, :D => :d2]") order!(net)
end
