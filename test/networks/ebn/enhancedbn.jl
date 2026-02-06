@testset "EnhancedBayesianNetwork" begin
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

    @testset "Structure" begin
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        @test net.A == sparse(zeros(length(nodes), length(nodes)))
        @test net.topology == Dict(:W => 1, :G => 2, :R => 3, :S => 4, :Rc => 5, :G2 => 6)
        @test net.nodes == nodes
        nodes = [weather, grass, grass, rain, sprinkler, rain2, grass2]
        @test_throws ErrorException("Invalid eBN: duplicate node names [:G]") EnhancedBayesianNetwork(nodes)
        rain3 = DiscreteNode(:R3, [:W])
        rain3[:W=>:sunny, :R3=>:yes] = 0.05
        rain3[:W=>:sunny, :R3=>:maybe] = 0.95
        rain3[:W=>:cloudy, :R3=>:yes] = 0.7
        rain3[:W=>:cloudy, :R3=>:maybe] = 0.3
        nodes = [weather, rain, rain3]
        @test_throws ErrorException("Invalid eBN: duplicate node states [:yes]") EnhancedBayesianNetwork(nodes)
    end

    @testset "add_child!" begin
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        @test_throws ErrorException("Invalid eBN: node 'W' have recursion") add_child!(net, weather, weather)
        @test_throws ErrorException("Invalid eBN: node G does not have the node(s) W in its CPT") add_child!(net, weather, grass)
        @test_throws ErrorException("Invalid eBN: node(s) [:G] are not functional node(s) and cannot be children of the continuous/functional node G2") add_child!(net, grass2, grass)
        @test_throws ErrorException("Invalid eBN: node(s) [:G] are not functional node(s) and cannot be children of the continuous/functional node Rc") add_child!(net, rain2, grass)
        add_child!(net, weather, [rain, sprinkler])
        @test net.A == sparse([1, 1], [3, 4], [true, true], 6, 6)
        add_child!(net, [rain, sprinkler], grass)
        @test net.A == sparse([1, 1, 3, 4], [3, 4, 2, 2], [true, true, true, true], 6, 6)
        add_child!(net, rain2, grass2)
        add_child!(net, sprinkler, grass2)
        @test net.A == sparse([1, 1, 3, 4, 4, 5], [3, 4, 2, 2, 6, 6], [true, true, true, true, true, true], 6, 6)
    end

    @testset "cyclicality connection parents and children" begin
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
    end

    @testset "verify parents, scenarios, exhaustiveness and functional" begin
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [sprinkler, rain2], grass2)
        add_child!(net, sprinkler, grass)
        @test_throws ErrorException("Invalid eBN: node G has node(s) '[:R]' defined in the CPT only, but they have not been added via add_child!") EnhancedBayesianNetworks.verify_parents(net, grass)
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

        @test_throws ErrorException("Invalid eBN: node G is missing the following scenario [:R => :yes, :S => :on, :G => :wet]") EnhancedBayesianNetworks.verify_scenarios(net, grass)
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

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [rain, rain2], grass2)

        @test_throws ErrorException("Invalid eBN: node R is a parent for the FuctionalNode G2 and cannot have an empty parameters attribute") EnhancedBayesianNetworks.verify_functional_parents(net, grass2)

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [sprinkler], grass2)
        @test_logs (:warn, "node G2 is a FunctionalNode with no continuous parents. Resulting failure probabilities are Boolean") EnhancedBayesianNetworks.verify_functional_parents(net, grass2)

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [rain2], grass2)
        @test_logs (:warn, "node G2 is a FunctionalNode with no discrete parents. Resulting eBN is a standard reliability analysis") EnhancedBayesianNetworks.verify_functional_parents(net, grass2)

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [rain2, sprinkler], grass2)
        @test isnothing(EnhancedBayesianNetworks.verify_functional_parents(net, grass2))
    end

    @testset "order!" begin
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
        @test_throws ErrorException("Invalid eBN: network is cyclic!") order!(net)

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [sprinkler, rain2], grass2)
        @test_throws ErrorException("Invalid eBN: network is not connected") order!(net)

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [sprinkler, rain2], grass2)
        add_child!(net, sprinkler, grass)
        @test_throws ErrorException("Invalid eBN: node G has node(s) '[:R]' defined in the CPT only, but they have not been added via add_child!") order!(net)

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
        @test_throws ErrorException("Invalid eBN: node G is missing the following scenario [:R => :yes, :S => :on, :G => :wet]") order!(net)

        grass[:R=>:yes, :S=>:on, :G=>:wet] = 1
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [sprinkler, rain2], grass2)
        add_child!(net, [rain, sprinkler], grass)
        @test isnothing(order!(net))

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
        @test_logs (:warn, "node G has CPT values 'Union{Real, Interval}[0, 0.999]' for the scenario [:R => :yes, :S => :on] and will be normalized!") order!(net)

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
        @test_throws ErrorException("Invalid CPT: node G has CPT values 'Union{Real, Interval}[0.3, 0.999]' not exhaustive and mutually exclusive for the scenario [:R => :yes, :S => :on]") order!(net)

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
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], [0.2, 0.3]]' for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") order!(net)

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
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], [0.8, 0.9]]' for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") order!(net)

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
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], 0.2]' for the scenario [:R => :yes, :S => :on], the sum of upper bound values must be greater than 1") order!(net)

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
        @test_throws ErrorException("Invalid CPT:  node G has CPT values 'Union{Real, Interval}[[0.3, 0.4], 0.8]' for the scenario [:R => :yes, :S => :on], the sum of lower bound values must be less than 1") order!(net)

        grass = DiscreteNode(:G, [:S, :R])
        grass[:R=>:yes, :S=>:on, :G=>:dry] = 0
        grass[:R=>:yes, :S=>:on, :G=>:wet] = 1
        grass[:R=>:yes, :S=>:off, :G=>:dry] = 0.05
        grass[:R=>:yes, :S=>:off, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:on, :G=>:dry] = 0.05
        grass[:R=>:no, :S=>:on, :G=>:wet] = 0.95
        grass[:R=>:no, :S=>:off, :G=>:dry] = 1
        grass[:R=>:no, :S=>:off, :G=>:wet] = 0
        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [rain, rain2], grass2)

        @test_throws ErrorException("Invalid eBN: node R is a parent for the FuctionalNode G2 and cannot have an empty parameters attribute") order!(net)

        nodes = [weather, grass, rain, sprinkler, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [sprinkler], grass2)
        @test_logs (:warn, "node G2 is a FunctionalNode with no continuous parents. Resulting failure probabilities are Boolean") order!(net)

        nodes = [rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, [rain2], grass2)
        @test_logs (:warn, "node G2 is a FunctionalNode with no discrete parents. Resulting eBN is a standard reliability analysis") order!(net)

        nodes = [weather, grass, rain, sprinkler, rain2, grass2]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [sprinkler, rain])
        add_child!(net, [rain, sprinkler], grass)
        add_child!(net, [rain2, sprinkler], grass2)
        @test isnothing(order!(net))

        nodes = [weather, grass, sprinkler, rain]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, weather, [rain, sprinkler])
        add_child!(net, [rain, sprinkler], grass)
        order!(net)
        @test issetequal(net.nodes, nodes)
        @test net.topology == Dict(:W => 1, :S => 2, :R => 3, :G => 4)
        @test net.A == sparse([1, 1, 2, 3], [2, 3, 4, 4], [true, true, true, true], 4, 4)
    end
end
