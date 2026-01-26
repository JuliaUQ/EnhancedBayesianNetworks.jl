@testset "Node Discretization" begin

    @testset "Format Intervals" begin
        discretization = ExactDiscretization([-10, 0, 10])
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root[] = truncated(Normal(0, 1), -10, 10)
        root = ContinuousNode(:z1, cpt_root, discretization)

        formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
        @test formatted_intervals == [[-10, 0.0], [0.0, 10]]

        discretization = ExactDiscretization([-9, 10])
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root[] = truncated(Normal(0, 1), -10, 10)
        root = ContinuousNode(:z1, cpt_root, discretization)

        @test_logs (:warn, "node z1 has minimum intervals value -9 > support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-11, 10])
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root[] = truncated(Normal(0, 1), -10, 10)
        root = ContinuousNode(:z1, cpt_root, discretization)

        @test_logs (:warn, "node z1 has minimum intervals value -11 < support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 9])
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root[] = truncated(Normal(0, 1), -10, 10)
        root = ContinuousNode(:z1, cpt_root, discretization)

        @test_logs (:warn, "node z1 has maximum intervals value 9 < support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 11])
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root[] = truncated(Normal(0, 1), -10, 10)
        root = ContinuousNode(:z1, cpt_root, discretization)

        @test_logs (:warn, "node z1 has maximum intervals value 11 > support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

        intervals = [[-Inf, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, Inf]]
        λ = 2

        exp1 = -Exponential(2) - 1
        exp2 = Exponential(2) + 1
        approx = [
            exp1,
            Uniform(-1, 0),
            Uniform(0.0, 1.0),
            exp2
        ]

        @test approx == EnhancedBayesianNetworks._approximate.(intervals, λ)

        dist = Normal()
        probs = [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696]

        @test all(isapprox.(EnhancedBayesianNetworks._discretize(dist, intervals), probs))
    end

    @testset "Root node" begin
        discretization = ExactDiscretization([0, Inf])
        cpt_root = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root[] = Normal()
        root = ContinuousNode(:z1, cpt_root, discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.name == :z1_d
        @test disc_root.cpt.data[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test disc_root.cpt.data[!, :Π] == [0.5, 0.5]

        @test cont_root.name == :z1
        @test cont_root.cpt.data[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test cont_root.cpt.data[!, :Π] == [truncated(Normal(), -Inf, 0.0), truncated(Normal(), 0.0, Inf)]

        discretization = ExactDiscretization([-Inf, 0, Inf])

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.name == :z1_d
        @test disc_root.cpt.data[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test disc_root.cpt.data[!, :Π] == [0.5, 0.5]

        @test cont_root.name == :z1
        @test cont_root.cpt.data[!, :z1_d] == [Symbol("[-Inf, 0.0]"), Symbol("[0.0, Inf]")]
        @test cont_root.cpt.data[!, :Π] == [truncated(Normal(), -Inf, 0.0), truncated(Normal(), 0.0, Inf)]

        discretization = ExactDiscretization([-1, 0, 1])
        cpt_root = ContinuousConditionalProbabilityTable{UnamedProbabilityBox}()
        cpt_root[] = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
        root = ContinuousNode(:z1, cpt_root, discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.cpt.data[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[-Inf, -1.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
        @test all(isapprox.(disc_root.cpt.data[!, :Π][1],
            (0.24173033745712885, 0.2901687869569368), atol=0.001))
        @test all(isapprox.(disc_root.cpt.data[!, :Π][2],
            (0.06680720126885804, 0.4012936743170763), atol=0.001))
        @test all(isapprox.(disc_root.cpt.data[!, :Π][3],
            (0.2417303374571288, 0.2901687869569368), atol=0.001))
        @test all(isapprox.(disc_root.cpt.data[!, :Π][4],
            (0.06680720126885809, 0.401293674317076), atol=0.001))

        dists = [
            UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], -Inf, -1.0),
            UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], -1.0, 0.0), UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], 0.0, 1.0),
            UnamedProbabilityBox{Normal}(Interval[Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)], 1.0, Inf)
        ]

        @test cont_root.cpt.data[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[-Inf, -1.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]

        cont_root.cpt.data[!, :Π] == dists

        discretization = ExactDiscretization([-1, 0, 1])
        cpt_root = ContinuousConditionalProbabilityTable{Tuple{<:Real,<:Real}}()
        cpt_root[] = (-1, 1)
        root = ContinuousNode(:z1, cpt_root, discretization)

        disc_root, cont_root = @suppress EnhancedBayesianNetworks._discretize(root)

        @test disc_root.cpt.data[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]")]

        @test all(isapprox.(disc_root.cpt.data[!, :Π][1],
            (0, 1); atol=0.001))

        @test all(isapprox.(disc_root.cpt.data[!, :Π][2],
            (0, 1); atol=0.001))

        dists = [
            (-1.0, 0.0),
            (0.0, 1.0)
        ]
        @test cont_root.cpt.data[!, :z1_d] == [Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]")]
        @test cont_root.cpt.data[!, :Π] == dists
    end

    @testset "Child node" begin
        λ = 1.5
        discretization = ApproximatedDiscretization([-Inf, -1, 0, 1, Inf], λ)

        cpt_child1 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}(:x)
        cpt_child1[:x=>:y] = Normal()
        cpt_child1[:x=>:n] = Normal(2.2)
        child1 = ContinuousNode(:β, cpt_child1, discretization)

        disc_child1, cont_child1 = EnhancedBayesianNetworks._discretize(child1)

        exp1 = -Exponential(λ) - 1
        exp2 = Exponential(λ) + 1
        @test cont_child1.cpt.data[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]
        @test cont_child1.cpt.data[!, :Π] == [Uniform(-1, 0), exp1, Uniform(0, 1.0), exp2]

        @test disc_child1.cpt.data[!, :x] == [:n, :n, :n, :n, :y, :y, :y, :y]
        @test disc_child1.cpt.data[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]"),
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]

        @test all(isapprox.(disc_child1.cpt.data[!, :Π], [
                0.013216309575582752,
                0.00068713793791584,
                0.1011662227082096,
                0.8849303297782918,
                0.341344746068543,
                0.15865525393145702,
                0.34134474606854304,
                0.15865525393145696,
            ]; atol=0.01))

        cpt_child2 = ContinuousConditionalProbabilityTable{UnamedProbabilityBox}(:x)
        cpt_child2[:x=>:y] = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
        cpt_child2[:x=>:n] = UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])
        child2 = ContinuousNode(:β, cpt_child2, discretization)

        disc_child2, cont_child2 = EnhancedBayesianNetworks._discretize(child2)

        @test cont_child2.cpt.data[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]
        @test cont_child2.cpt.data[!, :Π] == [Uniform(-1, 0), exp1, Uniform(0, 1), exp2]

        @test disc_child2.cpt.data[!, :x] == [:n, :n, :n, :n, :y, :y, :y, :y]

        @test disc_child2.cpt.data[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]"),
            Symbol("[-1.0, 0.0]"),
            Symbol("[-Inf, -1.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[1.0, Inf]")
        ]

        @test all(isapprox.(disc_child2.cpt.data[!, :Π][1],
            (0.021400233916549112, 0.14988228479452986); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][2],
            (0.001349898031630093, 0.15865525393145702); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][3],
            (0.1359051219832778, 0.19146246127401312); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][4],
            (0.5, 0.841344746068543); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][5],
            (0.24173033745712885, 0.2901687869569368); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][6],
            (0.06680720126885804, 0.4012936743170763); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][7],
            (0.2417303374571288, 0.2901687869569368); atol=0.01))
        @test all(isapprox.(disc_child2.cpt.data[!, :Π][8],
            (0.06680720126885809, 0.4012936743170763); atol=0.01))

        cpt_child3 = ContinuousConditionalProbabilityTable{Tuple{Real,Real}}(:x)
        cpt_child3[:x=>:y] = (-1, 0)
        cpt_child3[:x=>:n] = (0, 1)
        child3 = ContinuousNode(:β, cpt_child3, discretization)

        disc_child3, cont_child3 = @suppress EnhancedBayesianNetworks._discretize(child3)

        @test cont_child3.cpt.data[!, :β_d] == [Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]")]
        @test cont_child3.cpt.data[!, :Π] == [Uniform(-1, 0), Uniform(0, 1)]

        @test disc_child3.cpt.data[!, :x] == [:n, :n, :y, :y]
        @test disc_child3.cpt.data[!, :β_d] == [
            Symbol("[-1.0, 0.0]"),
            Symbol("[0.0, 1.0]"),
            Symbol("[-1.0, 0.0]"),
            Symbol("[0.0, 1.0]"),
        ]
        @test disc_child3.cpt.data[!, :Π] == [(0, 1), (0, 1), (0, 1), (0, 1)]
    end

    @testset "Network" begin
        cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
        cpt_root1[:x=>:y] = 0.2
        cpt_root1[:x=>:n] = 0.8
        root1 = DiscreteNode(:x, cpt_root1, Dict(:y => [Parameter(1, :x)], :n => [Parameter(0, :x), Parameter(5.6, :x1)]))

        cpt_root2 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:y)
        cpt_root2[:y=>:yes] = 0.4
        cpt_root2[:y=>:no] = 0.6
        root2 = DiscreteNode(:y, cpt_root2, Dict(:yes => [Parameter(2.2, :y)], :no => [Parameter(5.5, :y)]))

        discretization_root3 = ExactDiscretization([-Inf, 0, Inf])
        cpt_root3 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root3[] = Normal()
        root3 = ContinuousNode(:z1, cpt_root3, discretization_root3)

        standard1_name = :α
        standard1_states = DataFrame(:x => [:y, :y, :y, :y, :n, :n, :n, :n], :y => [:yes, :yes, :no, :no, :yes, :yes, :no, :no], :α => [:a, :b, :a, :b, :a, :b, :a, :b], :Π => [0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.5, 0.5]
        )
        standard1_parameters = Dict(:a => [Parameter(3, :α)], :b => [Parameter(10, :α)])
        standard1_node = DiscreteNode(standard1_name, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(standard1_states), standard1_parameters)

        standard2_name = :β
        standard2_states = DataFrame(:x => [:y, :n], :Π => [Normal(), Normal(2, 2)])
        discretization_standard2 = ApproximatedDiscretization([-Inf, 0.1, Inf], 1.5)
        standard2_node = ContinuousNode(standard2_name, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(standard2_states), discretization_standard2)

        functional2_name = :f2
        functional2_model = Model(df -> (df.α .^ 2 + df.z .^ 2) ./ 2, :value1)
        functional2_simulation = MonteCarlo(800)
        functional2_performance = df -> 1 .- 2 .* df.value1
        functional2_node = DiscreteFunctionalNode(functional2_name, [functional2_model], functional2_performance, functional2_simulation)

        nodes = [standard1_node, root1, root3, root2, standard2_node, functional2_node]
        net = EnhancedBayesianNetwork(nodes)

        add_child!(net, :x, :α)
        add_child!(net, :y, :α)
        add_child!(net, :x, :β)
        add_child!(net, :α, :f2)
        add_child!(net, :z1, :f2)
        order!(net)

        EnhancedBayesianNetworks._discretize!(net)
        d1, c1 = EnhancedBayesianNetworks._discretize(root3)
        d2, c2 = EnhancedBayesianNetworks._discretize(standard2_node)

        @test net.A == sparse([
            0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
        ])
        @test net.topology == Dict(:α => 4, :β_d => 6, :z1_d => 3, :y => 2, :z1 => 5, :f2 => 7, :β => 8, :x => 1)
        @test issetequal(net.nodes, [standard1_node, root1, d1, c1, root2, d2, c2, functional2_node])
    end
end