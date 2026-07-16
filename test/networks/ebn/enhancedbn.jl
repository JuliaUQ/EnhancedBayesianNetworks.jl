@testsnippet SetupSprinklereBN begin

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

end

@testitem "EnhancedBayesianNetwork - Struct" setup=[ExtraDeps, SetupSprinklereBN] begin
    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)
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

    ebn = EnhancedBayesianNetwork(DiscreteNode[])
    @test isempty(ebn.nodes)
    @test ebn.topology == Dict{Symbol,Int}()
    @test size(ebn.A) == (0, 0)
end

@testitem "EnhancedBayesianNetwork - add_child!" setup=[ExtraDeps, SetupSprinklereBN] begin
    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)
    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    @test_throws ErrorException("Invalid Network: nodes [:W] have a loop") add_child!(net, weather, weather)
    @test_throws ErrorException("Invalid Network: node :G does not have the node :W in its CPT") add_child!(net, weather, grass)
    @test_throws ErrorException("Invalid Network: nodes [:G] are not functional nodes and cannot be children of the continuous/functional node :G2") add_child!(net, grass2, grass)
    @test_throws ErrorException("Invalid Network: nodes [:G] are not functional nodes and cannot be children of the continuous/functional node :Rc") add_child!(net, rain2, grass)
    add_child!(net, weather, [rain, sprinkler])
    @test net.A == sparse([1, 1], [3, 4], [true, true], 6, 6)
    add_child!(net, [rain, sprinkler], grass)
    @test net.A == sparse([1, 1, 3, 4], [3, 4, 2, 2], [true, true, true, true], 6, 6)
    add_child!(net, rain2, grass2)
    add_child!(net, sprinkler, grass2)
    @test net.A == sparse([1, 1, 3, 4, 4, 5], [3, 4, 2, 2, 6, 6], [true, true, true, true, true, true], 6, 6)

    b = DiscreteNode(:b)
    b[:b=>:b1] = 0.5
    b[:b=>:b2] = 0.5
    @test_throws ErrorException("Invalid Network: nodes [:b] are not defined in the network") add_child!(net, b, sprinkler)
    @test_throws ErrorException("Invalid Network: nodes [:b] are not defined in the network") add_child!(net, :b, :S)
end

@testitem "EnhancedBayesianNetwork - Transmission" setup=[ExtraDeps, CheckSetup] begin
    parameters_root1 = [:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]]
    root1 = DiscreteNode(:x, parameters_root1)
    root1[:x=>:x1] = 0.3
    root1[:x=>:x2] = 0.7

    root2 = ContinuousNode(:y)
    root2[] = Normal()

    model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :fc)
    cont_functional = ContinuousFunctionalNode(:fc, [model1], MonteCarlo(300))

    model2 = Model(df -> df.fc .* 0.5, :fd)
    performance = df -> df.fc .- 0.5
    discrete_functional = DiscreteFunctionalNode(:fd, [model2], performance, MonteCarlo(300))

    nodes = [root1, root2, cont_functional, discrete_functional]

    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, root1, cont_functional)
    add_child!(net, root2, cont_functional)
    add_child!(net, cont_functional, discrete_functional)

    new_discrete_functional = DiscreteFunctionalNode(:fd, [model1, model2], performance, MonteCarlo(300))

    EnhancedBayesianNetworks._transfer_continuous_functional_node!(net, cont_functional)
    adj = sparse([1, 2], [3, 3], [true, true], 3, 3)
    @test net.A == adj
    @test issetequal([i.name for i in net.nodes], [:x, :y, :fd])
    @test net.topology == Dict(:x => 1, :y => 2, :fd => 3)
    @test discrete_functional.models == new_discrete_functional.models
    check_index_coherence(net)

    discrete_functional = DiscreteFunctionalNode(:fd, [model2], performance, MonteCarlo(300))
    discretization = ApproximatedDiscretization([-2, 0, 2], 2)
    cont_functional = ContinuousFunctionalNode(:fc, [model1], MonteCarlo(300), discretization)
    nodes = [root1, root2, cont_functional, discrete_functional]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, root1, cont_functional)
    add_child!(net, root2, cont_functional)
    add_child!(net, cont_functional, discrete_functional)
    @suppress order!(net)
    net1 = deepcopy(net)
    EnhancedBayesianNetworks._transfer_continuous_functional_node!(net, cont_functional)
    @test :fc ∈ [i.name for i in net.nodes]
    @test length(discrete_functional.models) == 1
    check_index_coherence(net)
end

@testitem "EnhancedBayesianNetwork - Discretize" setup=[ExtraDeps, CheckSetup] begin
    A = DiscreteNode(:A)
    A[:A=>:a1] = 0.2
    A[:A=>:a2] = 0.8

    discretization_B = ExactDiscretization([-1, 1])
    B = ContinuousNode(:B, discretization_B)
    B[] = Normal()

    discretization_C = ApproximatedDiscretization([-1, 1], 2)
    C = ContinuousNode(:C, [:A], discretization_C)
    C[:A=>:a1] = Interval(-1, 2)
    C[:A=>:a2] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))

    parameters_D = [:d1 => [Parameter(1, :D)], :d2 => [Parameter(2, :D)]]
    D = DiscreteNode(:D, parameters_D)
    D[:D=>:d1] = 0.2
    D[:D=>:d2] = 0.8

    model = Model(df -> df.C .- df.D .^ 2, :F_r)
    performance = df -> df.F_r
    simulation = MonteCarlo(20)
    F = DiscreteFunctionalNode(:F, model, performance, simulation)

    nodes = [A, B, C, D, F]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, A, C)
    add_child!(net, [C, B, D], F)
    disc1, cont1 = @suppress EnhancedBayesianNetworks._discretize(B)
    disc2, cont2 = @suppress EnhancedBayesianNetworks._discretize(C)
    discretized_net = @suppress EnhancedBayesianNetworks.discretize!(net)
    @test cont1.name ∈ [i.name for i in net.nodes]
    @test disc1.name ∈ [i.name for i in net.nodes]
    @test cont2.name ∈ [i.name for i in net.nodes]
    @test disc2.name ∈ [i.name for i in net.nodes]
    @test issetequal(parents(net, F), [:D, :B, :C])
    @test net.A == sparse([1, 2, 4, 5, 6, 7], [6, 3, 5, 3, 7, 3], [true, true, true, true, true, true], 7, 7)
    check_index_coherence(net)
end

@testitem "EnhancedBayesianNetwork - Markov Envelope" begin
    parameterY1 = [:yy1 => [Parameter(1, :y1)], :ny1 => [Parameter(0, :y1)]]
    Y1 = DiscreteNode(:y1, parameterY1)
    Y1[:y1=>:yy1] = 0.5
    Y1[:y1=>:ny1] = 0.5

    X1 = ContinuousNode(:x1)
    X1[] = Normal()

    X2 = ContinuousNode(:x2)
    X2[] = Normal()

    X3 = ContinuousNode(:x3)
    X3[] = Normal()

    model = Model(df -> df.y1 .+ df.x1, :y2)
    models = [model]
    simulation = MonteCarlo(200)
    performance = df -> df.y2
    Y2 = DiscreteFunctionalNode(:y2, models, performance, simulation)

    model = Model(df -> df.x1 .+ df.x2, :y3)
    models = [model]
    simulation = MonteCarlo(200)
    performance = df -> df.y3
    Y3 = DiscreteFunctionalNode(:y3, models, performance, simulation)

    model = Model(df -> df.x3 .+ df.x2, :y4)
    models = [model]
    simulation = MonteCarlo(200)
    performance = df -> df.y4
    Y4 = DiscreteFunctionalNode(:y4, models, performance, simulation)

    model = Model(df -> df.x3, :y5)
    models = [model]
    simulation = MonteCarlo(200)
    performance = df -> df.y5
    parameter = [:fail_y5 => [Parameter(1, :y5)], :fail_y5 => [Parameter(0, :y5)]]
    Y5 = DiscreteFunctionalNode(:y5, models, performance, simulation, parameter)

    model = Model(df -> df.y3, :x4)
    models = [model]
    simulation = MonteCarlo(200)
    X4 = ContinuousFunctionalNode(:x4, models, simulation)

    model = Model(df -> df.x4, :y6)
    models = [model]
    simulation = MonteCarlo(200)
    performance = df -> df.y6
    Y6 = DiscreteFunctionalNode(:y6, models, performance, simulation)
    nodes = [X1, X2, X3, Y1, Y2, Y3, Y4, Y5, X4, Y6]
    ebn = EnhancedBayesianNetwork(nodes)

    add_child!(ebn, :y1, :y2)
    add_child!(ebn, :x1, :y2)
    add_child!(ebn, :x1, :y3)
    add_child!(ebn, :x2, :y3)
    add_child!(ebn, :x2, :y4)
    add_child!(ebn, :x3, :y4)
    add_child!(ebn, :x3, :y5)
    add_child!(ebn, :y5, :x4)
    add_child!(ebn, :x4, :y6)

    @test issetequal(EnhancedBayesianNetworks._markov_continuous_group(ebn, X1), [X1, X2, X3])
    @test issetequal(EnhancedBayesianNetworks._markov_continuous_group(ebn, X2), [X1, X2, X3])
    @test issetequal(EnhancedBayesianNetworks._markov_continuous_group(ebn, X3), [X1, X2, X3])
    @test issetequal(EnhancedBayesianNetworks._markov_continuous_group(ebn, X4), [X4])

    envelopes = markov_envelope(ebn)
    @test issetequal(envelopes[1], [:y3, :y1, :x2, :x3, :y4, :y5, :y2, :x1])
    @test issetequal(envelopes[2], [:y6, :y5, :x4])
end

@testitem "EnhancedBayesianNetwork - verify functional parents" setup=[SetupSprinklereBN] begin
    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)
    rain3 = ContinuousNode(:R3, [:W])
    rain3[:W=>:sunny] = Normal()
    rain3[:W=>:cloudy] = Normal()
    nodes = [weather, grass, rain, sprinkler, rain2, rain3, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [sprinkler, rain2], grass2)
    add_child!(net, [rain, sprinkler], grass)
    @test isnothing(EnhancedBayesianNetworks._verify_parents(net, rain2))
    @test_throws ErrorException("Invalid CPT: node :R3 has nodes [:W] defined in the CPT only, but they have not been added via add_child!") EnhancedBayesianNetworks._verify_parents(net, rain3)
    add_child!(net, weather, rain3)
    @test isnothing(EnhancedBayesianNetworks._verify_parents(net, rain3))
    @test isnothing(EnhancedBayesianNetworks._verify_parents(net, grass2))

    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain, rain2], grass2)

    @test_throws ErrorException("Invalid Network: node :R is a parent for the FunctionalNode :G2 and cannot have an empty parameters attribute") EnhancedBayesianNetworks._verify_functional_parents(net, grass2)

    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [sprinkler], grass2)
    @test_logs (:warn, "Node :G2 is a FunctionalNode with no continuous parents. Resulting failure probabilities are Boolean") EnhancedBayesianNetworks._verify_functional_parents(net, grass2)

    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2], grass2)
    @test_logs (:warn, "Node :G2 is a FunctionalNode with no discrete parents. Resulting network is a standard reliability analysis") EnhancedBayesianNetworks._verify_functional_parents(net, grass2)

    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    @test isnothing(EnhancedBayesianNetworks._verify_functional_parents(net, grass2))
end

@testitem "EnhancedBayesianNetwork - verify ancestors & scenarios" setup=[SetupSprinklereBN] begin
    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)

    parameters_rain = [:yes => [Parameter(0, :R)], :no => [Parameter(1, :R)]]
    rain = DiscreteNode(:R, [:W], parameters_rain)
    rain[:W=>:sunny, :R=>:yes] = 0.05
    rain[:W=>:sunny, :R=>:no] = 0.95
    rain[:W=>:cloudy, :R=>:yes] = 0.7
    rain[:W=>:cloudy, :R=>:no] = 0.3

    rain2 = ContinuousNode(:Rc, [:W])
    rain2[:W=>:sunny] = Normal()
    rain2[:W=>:cloudy] = Normal()

    rain3 = ContinuousNode(:R3, [:W])
    rain3[:W=>:sunny] = Normal()
    rain3[:W=>:cloudy] = Normal()

    model = Model(df -> df.Rc .+ df.S, :G2)
    grass3 = ContinuousFunctionalNode(:G3, [:W, :S, :R], model)
    grass3[:W=>:sunny, :S=>:on, :R=>:yes] = MonteCarlo(10)
    grass3[:W=>:sunny, :S=>:off, :R=>:yes] = MonteCarlo(20)
    grass3[:W=>:cloudy, :S=>:on, :R=>:yes] = MonteCarlo(30)
    grass3[:W=>:cloudy, :S=>:off, :R=>:yes] = MonteCarlo(40)
    grass3[:W=>:sunny, :S=>:on, :R=>:no] = MonteCarlo(10)
    grass3[:W=>:sunny, :S=>:off, :R=>:no] = MonteCarlo(20)
    grass3[:W=>:cloudy, :S=>:on, :R=>:no] = MonteCarlo(30)
    grass3[:W=>:cloudy, :S=>:off, :R=>:no] = MonteCarlo(40)
    nodes = [weather, grass, rain, sprinkler, rain2, rain3, grass2, grass3]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain, rain2, rain3])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    @test_throws ErrorException("Invalid SimulationTable: node :G3 has nodes [:W, :S, :R] defined in the SimulationTable only, but they are not ancestors in the defined eBN") EnhancedBayesianNetworks._verify_ancestors(net, grass3)

    grass3 = ContinuousFunctionalNode(:G3, [:W], model)
    grass3[:W=>:sunny] = MonteCarlo(10)
    grass3[:W=>:cloudy] = MonteCarlo(40)
    nodes = [weather, grass, rain, sprinkler, rain2, rain3, grass2, grass3]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain, rain2, rain3])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    add_child!(net, [rain3, sprinkler], grass3)
    @test_throws ErrorException("Invalid SimulationTable: node :G3 has ancestors [:S] defined in the eBN only, but they are not present in its SimulationTable") EnhancedBayesianNetworks._verify_ancestors(net, grass3)

    grass3 = ContinuousFunctionalNode(:G3, [:W, :S], model)
    grass3[:W=>:sunny, :S=>:on] = MonteCarlo(10)
    grass3[:W=>:sunny, :S=>:off] = MonteCarlo(20)
    grass3[:W=>:cloudy, :S=>:on] = MonteCarlo(30)
    grass3[:W=>:cloudy, :S=>:off] = MonteCarlo(40)
    nodes = [weather, grass, rain, sprinkler, rain2, rain3, grass2, grass3]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain, rain2, rain3])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    add_child!(net, [rain3, sprinkler], grass3)
    @test isnothing(EnhancedBayesianNetworks._verify_ancestors(net, grass3))

    grass3 = ContinuousFunctionalNode(:G3, [:W, :S], model)
    grass3[:W=>:sunny, :S=>:on] = MonteCarlo(10)
    grass3[:W=>:sunny, :S=>:off] = MonteCarlo(20)
    nodes = [weather, grass, rain, sprinkler, rain2, rain3, grass2, grass3]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain, rain2, rain3])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    add_child!(net, [rain3, sprinkler], grass3)
    @test_throws ErrorException("Invalid SimulationTable: node :G3 is missing the following scenario [:W => :cloudy, :S => :on]") EnhancedBayesianNetworks._verify_scenarios(net, grass3)

    grass3 = ContinuousFunctionalNode(:G3, [:W, :S], model)
    grass3[:W=>:sunny, :S=>:on] = MonteCarlo(10)
    grass3[:W=>:sunny, :S=>:off] = MonteCarlo(20)
    grass3[:W=>:cloudy, :S=>:on] = MonteCarlo(30)
    grass3[:W=>:cloudy, :S=>:off] = MonteCarlo(40)
    nodes = [weather, grass, rain, sprinkler, rain2, rain3, grass2, grass3]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain, rain2, rain3])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)
    add_child!(net, [rain3, sprinkler], grass3)
    @test isnothing(EnhancedBayesianNetworks._verify_scenarios(net, grass3))
end

@testitem "EnhancedBayesianNetwork - build simulation table" setup=[ExtraDeps, SetupSprinklereBN] begin
    model = Model(df -> df.Rc .+ df.S, :G2)
    performance = df -> df.G2
    simulation = MonteCarlo(100)
    grass2 = DiscreteFunctionalNode(:G2, model, performance, simulation)

    nodes = [weather, grass, rain, sprinkler, rain2, grass2]
    net = EnhancedBayesianNetwork(nodes)
    add_child!(net, weather, [sprinkler, rain])
    add_child!(net, [rain, sprinkler], grass)
    add_child!(net, [rain2, sprinkler], grass2)

    EnhancedBayesianNetworks._build_simulations!(net, grass2)
    node = first(filter(n -> n.name == :G2, net.nodes))
    @test isa(node.simulation, EnhancedBayesianNetworks.ScenariosTable{EnhancedBayesianNetworks.DiscreteSimulation})
    @test issetequal(Symbol.(names(node.simulation.data)), [:S, :sim])
    @test issetequal(Symbol.(node.simulation.data.S), [:on, :off])
    @test issetequal(node.simulation.data.sim, [MonteCarlo(100), MonteCarlo(100)])

    st1 = node.simulation
    EnhancedBayesianNetworks._build_simulations!(net, node)   # 2nd call must be a no-op
    @test node.simulation === st1
end
