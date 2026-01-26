@testset "Common Networks Operations" begin

    weather_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:w)
    weather_cpt[:w=>:sunny] = 0.5
    weather_cpt[:w=>:cloudy] = 0.5
    weather = DiscreteNode(:w, weather_cpt)

    sprinkler_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:w, :s])
    sprinkler_cpt[:w=>:sunny, :s=>:on] = 0.9
    sprinkler_cpt[:w=>:sunny, :s=>:off] = 0.1
    sprinkler_cpt[:w=>:cloudy, :s=>:on] = 0.2
    sprinkler_cpt[:w=>:cloudy, :s=>:off] = 0.8
    sprinkler = DiscreteNode(:s, sprinkler_cpt)

    rain_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:w, :r])
    rain_cpt[:w=>:sunny, :r=>:no_rain] = 0.9
    rain_cpt[:w=>:sunny, :r=>:rain] = 0.1
    rain_cpt[:w=>:cloudy, :r=>:no_rain] = 0.8
    rain_cpt[:w=>:cloudy, :r=>:rain] = 0.2
    rain = DiscreteNode(:r, rain_cpt)

    grass_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:s, :r, :g])
    grass_cpt[:s=>:on, :r=>:no_rain, :g=>:dry] = 0.9
    grass_cpt[:s=>:on, :r=>:no_rain, :g=>:wet] = 0.1
    grass_cpt[:s=>:on, :r=>:rain, :g=>:dry] = 0.9
    grass_cpt[:s=>:on, :r=>:rain, :g=>:wet] = 0.1
    grass_cpt[:s=>:off, :r=>:no_rain, :g=>:dry] = 0.9
    grass_cpt[:s=>:off, :r=>:no_rain, :g=>:wet] = 0.1
    grass_cpt[:s=>:off, :r=>:rain, :g=>:dry] = 0.9
    grass_cpt[:s=>:off, :r=>:rain, :g=>:wet] = 0.1
    grass = DiscreteNode(:g, grass_cpt)

    @testset "add_child!" begin

        nodes = [weather, grass, rain, sprinkler]
        net = EnhancedBayesianNetwork(nodes)
        topology = Dict(:w => 1, :s => 4, :g => 2, :r => 3)
        A = spzeros(4, 4)
        @test net.topology == topology
        @test issetequal(net.nodes, nodes)
        @test net.A == A

        @test_throws ErrorException("Recursion on the same node 'w' is not allowed in EnhancedBayesianNetworks") add_child!(net, :w, :w)
        @test_throws ErrorException("node 'w' is a root node and cannot have parents") add_child!(net, :s, :w)
        rain_state1 = DataFrame(:a => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Π => [0.9, 0.1, 0.2, 0.8])
        rain1 = DiscreteNode(:r, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(rain_state1))
        nodes1 = [weather, grass, rain1, sprinkler]
        net1 = EnhancedBayesianNetwork(nodes1)
        @test_throws ErrorException("trying to set node 'r' as child of node 'w', but 'r' has a cpt that does not contains 'w' in the scenarios: $(rain1.cpt)") @suppress add_child!(net1, weather, rain1)
        rain_state2 = DataFrame(:w => [:sunny, :sunny, :cloudies, :cloudies], :r => [:no_rain, :rain, :no_rain, :rain], :Π => [0.9, 0.1, 0.2, 0.8])
        rain2 = DiscreteNode(:r, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(rain_state2))
        nodes2 = [weather, grass, rain2, sprinkler]
        net2 = EnhancedBayesianNetwork(nodes2)
        @test_throws ErrorException("child node 'r' has scenarios [:cloudies, :sunny], that is not coherent with its parent node 'w' with states [:cloudy, :sunny]") add_child!(net2, weather, rain2)

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass1 = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes4 = [weather, rain, sprinkler, grass1]
        net4 = EnhancedBayesianNetwork(nodes4)
        @test_throws ErrorException("functional node 'g' can have only functional children. 'r' is not a functional node") add_child!(net4, :g, :r)
        net_new1 = deepcopy(net)
        net_new2 = deepcopy(net)
        add_child!(net, weather, rain)
        A_net = sparse([
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0
        ])
        @test net.topology == topology
        @test net.nodes == nodes
        @test net.A == A_net
        add_child!(net_new1, :w, :r)
        @test net_new1 == net
        add_child!(net_new2, 1, 3)
        add_child!(net_new2, :w, :r)
        @test net_new2 == net
    end

    @testset "parents & children" begin

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes = [weather, rain, sprinkler, grass]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :r)
        add_child!(net, :w, :s)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)

        @test parents(net, :w) == (Int64[], Symbol[], AbstractNode[])
        @test parents(net, 1) == (Int64[], Symbol[], AbstractNode[])
        @test parents(net, weather) == (Int64[], Symbol[], AbstractNode[])
        @test parents(net, :g) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test parents(net, 4) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test parents(net, grass) == ([2, 3], [:r, :s], [rain, sprinkler])

        @test children(net, :w) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test children(net, 1) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test children(net, weather) == ([2, 3], [:r, :s], [rain, sprinkler])
        @test children(net, :g) == (Int64[], Symbol[], AbstractNode[])
        @test children(net, 4) == (Int64[], Symbol[], AbstractNode[])
        @test children(net, grass) == (Int64[], Symbol[], AbstractNode[])
    end

    @testset "discrete_ancestors" begin
        cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:X1)
        cpt_root1[:X1=>:y] = 0.2
        cpt_root1[:X1=>:n] = 0.8
        root1 = DiscreteNode(:X1, cpt_root1)

        cpt_root2 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:X2)
        cpt_root2[:X2=>:yes] = 0.4
        cpt_root2[:X2=>:no] = 0.6
        root2 = DiscreteNode(:X2, cpt_root2, Dict(:yes => [Parameter(2.2, :X2)], :no => [Parameter(5.5, :X2)]))

        cpt_root3 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root3[] = Normal()
        root3 = ContinuousNode(:Y1, cpt_root3, ExactDiscretization([0, 0.2, 1]))

        child1_states = DataFrame(:X1 => [:y, :y, :n, :n], :C1 => [:c1y, :c1n, :c1y, :c1n], :Π => [0.3, 0.7, 0.4, 0.6])
        child1 = DiscreteNode(:C1, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(child1_states), Dict(:c1y => [Parameter(1, :X1)], :c1n => [Parameter(0, :X1)]))

        child2 = ContinuousNode(:C2, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:X2 => [:yes, :no], :Π => [Normal(), Normal(1, 1)])))

        disc_D = ApproximatedDiscretization([-1.1, 0, 0.11], 2)
        model1 = [Model(df -> (df.X1 .^ 2) ./ 2 .- df.C2, :fun1)]
        simulation1 = MonteCarlo(300)
        functional1_node = ContinuousFunctionalNode(:F1, model1, simulation1, disc_D)

        net = EnhancedBayesianNetwork([root1, root2, root3, child1, child2, functional1_node])
        add_child!(net, root1, child1)
        add_child!(net, root2, child2)
        add_child!(net, child1, functional1_node)
        add_child!(net, child2, functional1_node)

        @test issetequal(discrete_ancestors(net, functional1_node), [root2, child1])
        @test isempty(discrete_ancestors(net, root1))
        th_scenarios = [
            Dict(:C1 => :c1n, :X2 => :no)
            Dict(:C1 => :c1y, :X2 => :no)
            Dict(:C1 => :c1n, :X2 => :yes)
            Dict(:C1 => :c1y, :X2 => :yes)]
        @test EnhancedBayesianNetworks._theoretical_scenarios(net, functional1_node) == th_scenarios
        @test EnhancedBayesianNetworks._theoretical_scenarios(net, root1) == [Dict()]
    end

    @testset "verify net" begin

        grass_model = Model(df -> df.r .+ df.s, :g)
        grass_simulation = MonteCarlo(200)
        grass_performance = df -> df.g
        grass = DiscreteFunctionalNode(:g, [grass_model], grass_performance, grass_simulation)
        nodes = [weather, rain, sprinkler, grass]
        net = EnhancedBayesianNetwork(nodes)
        @test isnothing(EnhancedBayesianNetworks._verify_child_node(net, weather))
        @test_throws ErrorException("node 'r''s cpt requires exctly the nodes '[:w]' to be its parents, but provided parents are 'Symbol[]'") EnhancedBayesianNetworks._verify_child_node(net, rain)
        @test_throws ErrorException("node 'r''s cpt requires exctly the nodes '[:w]' to be its parents, but provided parents are 'Symbol[]'") EnhancedBayesianNetworks._verify_net(net)

        add_child!(net, :w, :r)
        add_child!(net, :w, :s)

        @test_throws ErrorException("functional node 'g' must have at least one parent") EnhancedBayesianNetworks._verify_functional_node(net, grass)
        @test_throws ErrorException("functional node 'g' must have at least one parent") EnhancedBayesianNetworks._verify_net(net)

        add_child!(net, :s, :g)
        add_child!(net, :r, :g)

        @test_throws ErrorException("node 's' is a discrete parent of a functional node and cannot have an empty parameters vector") EnhancedBayesianNetworks._non_empty_parameters_vector(net, sprinkler)
        @test_throws ErrorException("node 'r' is a discrete parent of a functional node and cannot have an empty parameters vector") EnhancedBayesianNetworks._non_empty_parameters_vector(net, rain)
        @suppress @test_throws ErrorException("node 'r' is a discrete parent of a functional node and cannot have an empty parameters vector") EnhancedBayesianNetworks._verify_net(net)

        sprinkler_states = DataFrame(
            :w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Π => [0.9, 0.1, 0.2, 0.8]
        )
        sprinkler2 = DiscreteNode(:s, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(sprinkler_states), Dict(:on => [Parameter(1, :S)], :off => [Parameter(2, :S)]))
        nodes2 = [weather, rain, sprinkler2, grass]
        net2 = EnhancedBayesianNetwork(nodes2)
        add_child!(net2, weather, sprinkler2)
        add_child!(net2, weather, rain)
        add_child!(net2, sprinkler2, grass)
        @test isnothing(EnhancedBayesianNetworks._non_empty_parameters_vector(net, sprinkler2))
        @test_logs (:warn, "functional node 'g' have no continuous parents. All the simulations will return the same output") EnhancedBayesianNetworks._verify_functional_node(net2, grass)
        @test_logs (:warn, "functional node 'g' have no continuous parents. All the simulations will return the same output") EnhancedBayesianNetworks._verify_net(net2)
        @suppress isnothing(EnhancedBayesianNetworks._verify_functional_node(net2, grass))
        @suppress isnothing(EnhancedBayesianNetworks._verify_net(net2))

        nodes3 = [weather, rain, sprinkler, grass]
        net3 = EnhancedBayesianNetwork(nodes3)
        @test_throws ErrorException("functional node 'g' must have at least one parent") EnhancedBayesianNetworks._verify_functional_node(net3, grass)
        add_child!(net2, sprinkler2, grass)
        @test_logs (:warn, "functional node 'g' have no continuous parents. All the simulations will return the same output") EnhancedBayesianNetworks._verify_functional_node(net2, grass)
        @suppress isnothing(EnhancedBayesianNetworks._verify_functional_node(net2, grass)
        )

        sprinkler_states3 = DataFrame(
            :w => [:sunny, :sunny, :cloudy, :cloudy, :t, :t, :not_t, :not_t], :s => [:on, :off, :on, :off, :on, :off, :on, :off], :Π => [0.9, 0.1, 0.2, 0.8, 0.1, 0.9, 0.4, 0.6]
        )
        sprinkler3 = DiscreteNode(:s, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(sprinkler_states3))
        net3 = EnhancedBayesianNetwork([weather, rain, sprinkler3, grass])
        add_child!(net3, :w, :r)
        add_child!(net3, :w, :s)
        add_child!(net3, :s, :g)
        add_child!(net3, :r, :g)
        @test_throws ErrorException("node 's' has defined cpt scenarios $(sprinkler3.cpt) not coherent with the theoretical one [Dict(:w => :cloudy), Dict(:w => :sunny)]") EnhancedBayesianNetworks._verify_child_node(net3, sprinkler3)
        @test_throws ErrorException("node 's' has defined cpt scenarios $(sprinkler3.cpt) not coherent with the theoretical one [Dict(:w => :cloudy), Dict(:w => :sunny)]") EnhancedBayesianNetworks._verify_net(net3)
    end

    @testset "order network" begin
        cpt_root = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
        cpt_root[:A=>:a1] = 0.5
        cpt_root[:A=>:a2] = 0.5
        root = DiscreteNode(:A, cpt_root)

        child1 = DiscreteNode(:B, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:D => [:d1, :d2, :d1, :d2, :d1, :d2, :d1, :d2], :A => [:a1, :a1, :a2, :a2, :a1, :a1, :a2, :a2], :B => [:b1, :b2, :b1, :b2, :b1, :b2, :b1, :b2], :Π => [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])))

        child2 = DiscreteNode(:C, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:B => [:b1, :b1, :b2, :b2], :C => [:c1, :c2, :c1, :c2], :Π => [0.5, 0.5, 0.5, 0.5])))

        child3 = DiscreteNode(:D, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:C => [:c1, :c1, :c2, :c2], :D => [:d1, :d2, :d1, :d2], :Π => [0.5, 0.5, 0.5, 0.5])))

        grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Π => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        grass = DiscreteNode(:g, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(grass_states))

        nodes = [weather, sprinkler, rain, grass]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :s)
        add_child!(net, :w, :r)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)

        order!(net)
        @test net.A == sparse(Matrix([0 1.0 1.0 0; 0 0 0 1.0; 0 0 0 1.0; 0 0 0 0]))
        @test net.topology == Dict(:w => 1, :s => 2, :g => 4, :r => 3)
        @test net.nodes == [weather, sprinkler, rain, grass]
        @test isnothing(EnhancedBayesianNetworks._verify_net(net))
    end

    @testset "add & remove nodes" begin
        sprinkler_states = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :s => [:on, :off, :on, :off], :Π => [0.9, 0.1, 0.2, 0.8])
        sprinkler = DiscreteNode(:s, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(sprinkler_states))
        rain_state = DataFrame(:w => [:sunny, :sunny, :cloudy, :cloudy], :r => [:no_rain, :rain, :no_rain, :rain], :Π => [0.9, 0.1, 0.2, 0.8])
        rain = DiscreteNode(:r, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(rain_state))
        grass_states = DataFrame(:s => [:on, :on, :on, :on, :off, :off, :off, :off], :r => [:no_rain, :no_rain, :rain, :rain, :no_rain, :no_rain, :rain, :rain], :g => [:dry, :wet, :dry, :wet, :dry, :wet, :dry, :wet], :Π => [0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1])
        grass = DiscreteNode(:g, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(grass_states))

        nodes = [weather, sprinkler, rain, grass]
        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, :w, :s)
        add_child!(net, :w, :r)
        add_child!(net, :s, :g)
        add_child!(net, :r, :g)
        order!(net)
        net1 = deepcopy(net)
        net2 = deepcopy(net)
        net3 = deepcopy(net)

        EnhancedBayesianNetworks._remove_node!(net1, grass)
        EnhancedBayesianNetworks._remove_node!(net2, :g)
        EnhancedBayesianNetworks._remove_node!(net3, 4)

        @test issetequal(net1.nodes, [weather, sprinkler, rain])
        adj = [
            0.0 1.0 1.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0
        ]
        @test net1.A == adj
        @test net1.topology == Dict(:w => 1, :r => 3, :s => 2)
        @test net2 == net1
        @test net3 == net1

        net4 = deepcopy(net1)
        net5 = deepcopy(net1)
        net6 = deepcopy(net1)

        EnhancedBayesianNetworks._add_node!(net4, grass)
        EnhancedBayesianNetworks._add_node!(net5, grass)
        EnhancedBayesianNetworks._add_node!(net6, grass)

        add_child!(net4, :s, :g)
        add_child!(net4, :r, :g)
        order!(net4)
        add_child!(net5, :s, :g)
        add_child!(net5, :r, :g)
        order!(net5)
        add_child!(net6, :s, :g)
        add_child!(net6, :r, :g)
        order!(net6)

        @test net4 == net
        @test net5 == net
        @test net6 == net
    end
end