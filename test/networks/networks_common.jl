@testset "Networks Common (Only eBN, must be expanded)" begin
    weather = DiscreteNode(:W)
    weather[:W=>:sunny] = 0.5
    weather[:W=>:cloudy] = 0.5

    sprinkler_parameter = [:on => [Parameter(0.5, :S)], :off => [Parameter(0, :S)]]
    sprinkler = DiscreteNode(:S, [:W], sprinkler_parameter)
    sprinkler[:W=>:sunny, :S=>:on] = 0.7
    sprinkler[:W=>:sunny, :S=>:off] = 0.3
    sprinkler[:W=>:cloudy, :S=>:on] = 0.05
    sprinkler[:W=>:cloudy, :S=>:off] = 0.95

    rain = DiscreteNode(:R, [:W])
    rain[:W=>:sunny, :R=>:yes] = 0.05
    rain[:W=>:sunny, :R=>:no] = 0.95
    rain[:W=>:cloudy, :R=>:yes] = 0.7
    rain[:W=>:cloudy, :R=>:no] = 0.3

    rain2 = ContinuousNode(:Rc)
    rain2[] = Normal()

    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
    grass[:R=>:yes, :S=>:on, :G=>:wet] = 1
    grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
    grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
    grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
    grass[:R=>:no, :S=>:off, :G=>:dry] = 1
    grass[:R=>:no, :S=>:off, :G=>:wet] = 0

    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)

    @testset "cyclicality connection parents children and ancestors" begin
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
        net = EnhancedBayesianNetwork([A, B, weather])
        add_child!(net, A, B)
        add_child!(net, B, A)
        @test EnhancedBayesianNetworks.iscyclic(net)
        @test !EnhancedBayesianNetworks.isconnected(net)

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

        rain2 = ContinuousNode(:Rc, [:W])
        rain2[:W=>:sunny] = Normal()
        rain2[:W=>:cloudy] = Normal()
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain, rain2])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [rain2, sprinkler], grass2)
        order!(net)
        @test issetequal(EnhancedBayesianNetworks.ancestors(net, grass2), [:W, :S])
        @test issetequal(EnhancedBayesianNetworks.ancestors(net, :G2), [:W, :S])
    end

    @testset "verify parents, scenarios, exhaustiveness and functional" begin
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [sprinkler, rain2], grass2)
        add_child!(net, sprinkler, grass)
        @test_throws ErrorException("Invalid CPT: node G has node(s) '[:R]' defined in the CPT only, but they have not been added via add_child!") EnhancedBayesianNetworks.verify_parents(net, grass)
        add_child!(net, rain, grass)
        @test isnothing(EnhancedBayesianNetworks.verify_parents(net, grass))
        @test isnothing(EnhancedBayesianNetworks.verify_parents(net, grass2))

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)

        @test_throws ErrorException("Invalid CPT: node G is missing the following scenario [:R => :yes, :S => :on, :G => :wet]") EnhancedBayesianNetworks.verify_scenarios(net, grass)
        grass[:R=>:yes, :S=>:on, :G=>:wet] = 1
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [sprinkler, rain2], grass2)
        add_child!(net, [rain, sprinkler], grass)
        @test isnothing(EnhancedBayesianNetworks.verify_scenarios(net, grass))

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
        grass[:R=>:yes, :S=>:on, :G=>:wet] = 0.999
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        @test_logs (:warn, "node G has CPT values 'Union{Real, Interval}[0, 0.999]' for the scenario [:R => :yes, :S => :on] and will be normalized!") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass)
        @test filter(grass.cpt, ([:S, :R, :G] .=> [:on, :yes, :wet])...).Π == [1.0]

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = 0.3
        grass[:R=>:yes, :S=>:on, :G=>:wet] = 0.999
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        @test_throws ErrorException("Invalid CPT: node G has CPT values 'Union{Real, Interval}[0.3, 0.999]' not exhaustive and mutually exclusive for the scenario [:R => :yes, :S => :on]") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass)

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
        grass[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.2, 0.3)
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], [0.2, 0.3]]' for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass)

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
        grass[:R=>:yes, :S=>:on, :G=>:wet] = Interval(0.8, 0.9)
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], [0.8, 0.9]]' for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass)

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
        grass[:R=>:yes, :S=>:on, :G=>:wet] = 0.2
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], 0.2]' for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass)

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = Interval(0.3, 0.4)
        grass[:R=>:yes, :S=>:on, :G=>:wet] = 0.8
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], 0.8]' for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") EnhancedBayesianNetworks.verify_exhaustiveness(net, grass)
    end

    @testset "Markov Blanket" begin
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

        nodes = [x1, x2, x4, x8, x5, x7, x11, x3, x6, x9, x10, x12, x13]
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
        order!(net)

        @test issetequal(markov_blanket(net, :x6), [:x3, :x4, :x5, :x8, :x9, :x10])
        @test issetequal(markov_blanket(net, x6), [:x3, :x4, :x5, :x8, :x9, :x10])
    end

    # @testset "add & remove nodes" begin
    #     sprinkler_states = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Π => [0.9, 0.1, 0.2, 0.8])
    #     sprinkler = DiscreteNode(:s, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(sprinkler_states))
    #     rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Π => [0.9, 0.1, 0.2, 0.8])
    #     rain = DiscreteNode(:r, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(rain_state))
    #     grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Π => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
    #     grass = DiscreteNode(:g, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(grass_states))

    #     nodes = [weather, sprinkler, rain, grass]
    #     net = EnhancedBayesianNetwork(nodes)
    #     add_child!(net, :w, :s)
    #     add_child!(net, :w, :r)
    #     add_child!(net, :s, :g)
    #     add_child!(net, :r, :g)
    #     order!(net)
    #     net1 = deepcopy(net)
    #     net2 = deepcopy(net)
    #     net3 = deepcopy(net)

    #     EnhancedBayesianNetworks._remove_node!(net1, grass)
    #     EnhancedBayesianNetworks._remove_node!(net2, :g)
    #     EnhancedBayesianNetworks._remove_node!(net3, 4)

    #     @test issetequal(net1.nodes, [weather, sprinkler, rain])
    #     adj = [
    #         0.0 1.0 1.0;
    #         0.0 0.0 0.0;
    #         0.0 0.0 0.0
    #     ]
    #     @test net1.A == adj
    #     @test net1.topology == Dict(:w => 1, :r => 3, :s => 2)
    #     @test net2 == net1
    #     @test net3 == net1

    #     net4 = deepcopy(net1)
    #     net5 = deepcopy(net1)
    #     net6 = deepcopy(net1)

    #     EnhancedBayesianNetworks._add_node!(net4, grass)
    #     EnhancedBayesianNetworks._add_node!(net5, grass)
    #     EnhancedBayesianNetworks._add_node!(net6, grass)

    #     add_child!(net4, :s, :g)
    #     add_child!(net4, :r, :g)
    #     order!(net4)
    #     add_child!(net5, :s, :g)
    #     add_child!(net5, :r, :g)
    #     order!(net5)
    #     add_child!(net6, :s, :g)
    #     add_child!(net6, :r, :g)
    #     order!(net6)

    #     @test net4 == net
    #     @test net5 == net
    #     @test net6 == net
    # end
end