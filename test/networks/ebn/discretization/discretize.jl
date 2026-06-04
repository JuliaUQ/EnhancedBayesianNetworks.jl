@testset "Node Discretization" begin

    @testset "Format Intervals" begin
        discretization = ExactDiscretization([-10, 0, 10])
        root = ContinuousNode(:z1, discretization)
        root[] = truncated(Normal(), -10, 10)

        formatted_intervals = EnhancedBayesianNetworks._format_interval(root)
        @test formatted_intervals == [[-10, 0.0], [0.0, 10]]

        discretization = ExactDiscretization([-9, 10])
        root = ContinuousNode(:z1, discretization)
        root[] = truncated(Normal(), -10, 10)

        @test_logs (:warn, "node :z1 has minimum intervals value -9 > support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-11, 10])
        root = ContinuousNode(:z1, discretization)
        root[] = truncated(Normal(), -10, 10)

        @test_logs (:warn, "node :z1 has minimum intervals value -11 < support lower bound -10.0. Lower bound will be used as intervals start") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 9])
        root = ContinuousNode(:z1, discretization)
        root[] = truncated(Normal(), -10, 10)

        @test_logs (:warn, "node :z1 has maximum intervals value 9 < support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

        discretization = ExactDiscretization([-10, 11])
        root = ContinuousNode(:z1, discretization)
        root[] = truncated(Normal(), -10, 10)

        @test_logs (:warn, "node :z1 has maximum intervals value 11 > support upper bound 10.0. Upper bound will be used as intervals end") EnhancedBayesianNetworks._format_interval(root)

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
    end

    @testset "discretize distributions" begin
        intervals = [[-Inf, -1.0], [-1.0, 0.0], [0.0, 1.0], [1.0, Inf]]
        dist = Normal()
        probs = map(i -> EnhancedBayesianNetworks._discretize(dist, i), intervals)
        check = [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696]
        @test all(isapprox.(probs, check, atol=0.001))

        dist = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
        probs = map(i -> EnhancedBayesianNetworks._discretize(dist, i), intervals)
        check = [Interval(0.022750131948179205, 0.15865525393145702), Interval(0.1359051219832778, 0.341344746068543), Interval(0.341344746068543, 0.34134474606854304), Interval(0.15865525393145696, 0.5)]
        @test probs[1] == check[1]
        @test probs[2] == check[2]
        @test probs[3] == check[3]
        @test probs[4] == check[4]

        intervals = [[-1.0, 0.0], [0.0, 1.0]]
        dist = Interval(-1, 1)
        probs = map(i -> EnhancedBayesianNetworks._discretize(dist, i), intervals)
        check = [Interval(0, 1), Interval(0, 1)]
        @test probs[1] == check[1]
        @test probs[2] == check[2]
    end

    @testset "approximate" begin
        intervals = [[-Inf, -1.0], [-1.0, 1.0], [1.0, Inf]]
        λ = 2
        approximated_dist = EnhancedBayesianNetworks._approximate.(intervals, λ)
        @test approximated_dist[1].μ == -1
        @test approximated_dist[1].σ == -1
        @test isa(approximated_dist[1].ρ, Exponential)
        @test approximated_dist[1].ρ.θ == λ
        @test approximated_dist[2].a == -1
        @test approximated_dist[2].b == 1
        @test isa(approximated_dist[2], Uniform)
        @test approximated_dist[3].μ == 1
        @test approximated_dist[3].σ == 1
        @test isa(approximated_dist[3].ρ, Exponential)
        @test approximated_dist[3].ρ.θ == λ
    end

    @testset "discretize node" begin
        @testset "Root nodes" begin
            discretization = ExactDiscretization([-1, 0, 1])
            node = ContinuousNode(:x, discretization)
            node[] = Normal()
            discretized_node, new_continuous = @suppress EnhancedBayesianNetworks._discretize(node)
            discretized_states = [Symbol("[-Inf, -1.0]"), Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
            @test discretized_node.name == Symbol(string(node.name) * "_d")
            @test isempty(discretized_node.parameters)
            @test isnothing(discretized_node.results)
            @test Symbol.(names(discretized_node.cpt.data)) == [:x_d, :Π]
            @test discretized_node.cpt.data.x_d == discretized_states
            @test isapprox([discretized_node[discretized_node.name.=>i] for i in discretized_states], [0.15865525393145702, 0.341344746068543, 0.34134474606854304, 0.15865525393145696], atol=0.001)
            @test new_continuous.name == node.name
            @test isempty(new_continuous.discretization)
            @test Symbol.(names(new_continuous.cpt.data)) == [:x_d, :Π]
            @test new_continuous.cpt.data.x_d == discretized_states
            @test new_continuous.cpt.data.Π == [
                truncated(Normal(); lower=-Inf, upper=-1.0),
                truncated(Normal(); lower=-1.0, upper=0.0),
                truncated(Normal(); lower=0.0, upper=1.0),
                truncated(Normal(); lower=1.0, upper=Inf)
            ]

            node = ContinuousNode(:x, discretization)
            node[] = ProbabilityBox{Normal}(Dict(:μ => Interval(0, 1), :σ => 1))
            discretized_node, new_continuous = @suppress EnhancedBayesianNetworks._discretize(node)
            discretized_states = [Symbol("[-Inf, -1.0]"), Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
            @test discretized_node.name == Symbol(string(node.name) * "_d")
            @test isempty(discretized_node.parameters)
            @test isnothing(discretized_node.results)
            @test Symbol.(names(discretized_node.cpt.data)) == [:x_d, :Π]
            @test discretized_node.cpt.data.x_d == discretized_states
            @test [discretized_node[discretized_node.name.=>i] for i in discretized_states] == [
                Interval(0.022750131948179205, 0.15865525393145702),
                Interval(0.1359051219832778, 0.341344746068543),
                Interval(0.341344746068543, 0.34134474606854304),
                Interval(0.15865525393145696, 0.5)
            ]
            @test new_continuous.name == node.name
            @test isempty(new_continuous.discretization)
            @test Symbol.(names(new_continuous.cpt.data)) == [:x_d, :Π]
            @test new_continuous.cpt.data.x_d == discretized_states
            @test [new_continuous.cpt.data.Π[i].parameters for i in range(1, 4)] == [
                Dict(:μ => Interval(0, 1), :σ => 1),
                Dict(:μ => Interval(0, 1), :σ => 1),
                Dict(:μ => Interval(0, 1), :σ => 1),
                Dict(:μ => Interval(0, 1), :σ => 1)
            ]
            @test [new_continuous.cpt.data.Π[i].lb for i in range(1, 4)] == [-Inf, -1.0, 0.0, 1.0]
            @test [new_continuous.cpt.data.Π[i].ub for i in range(1, 4)] == [-1.0, 0.0, 1.0, Inf]
        end

        @testset "Child nodes" begin
            discretization = ApproximatedDiscretization([-1, 0, 1], 2)
            node = ContinuousNode(:x, [:y, :z], discretization)
            node[:y=>:y1, :z=>:z1] = Normal()
            node[:y=>:y1, :z=>:z2] = Normal(2, 2)
            node[:y=>:y2, :z=>:z1] = Interval(0, 3)
            node[:y=>:y2, :z=>:z2] = Normal(4, 4)
            discretized_states = [Symbol("[-Inf, -1.0]"), Symbol("[-1.0, 0.0]"), Symbol("[0.0, 1.0]"), Symbol("[1.0, Inf]")]
            discretized_node, new_continuous = @suppress EnhancedBayesianNetworks._discretize(node)
            @test discretized_node.name == Symbol(string(node.name) * "_d")
            @test isempty(discretized_node.parameters)
            @test isnothing(discretized_node.results)
            @test Symbol.(names(discretized_node.cpt.data)) == [:y, :z, :x_d, :Π]
            @test unique(discretized_node.cpt.data.x_d) == discretized_states
            @test discretized_node.cpt.data.Π == [
                0.15865525393145702,
                0.06680720126885804,
                Interval(0, 1),
                0.10564977366685525,
                0.341344746068543,
                0.09184805266259898,
                Interval(0, 1),
                0.053005480264601765,
                0.34134474606854304,
                0.14988228479452986,
                Interval(0, 1),
                0.06797209844541116,
                0.15865525393145696,
                0.6914624612740131,
                Interval(0, 1),
                0.7733726476231318
            ]

            @test new_continuous.name == node.name
            @test isempty(new_continuous.discretization)
            @test Symbol.(names(new_continuous.cpt.data)) == [:x_d, :Π]
            @test unique(new_continuous.cpt.data.x_d) == discretized_states
            @test isa(new_continuous.cpt.data.Π[1].ρ, Exponential)
            @test isa(new_continuous.cpt.data.Π[2], Uniform)
            @test isa(new_continuous.cpt.data.Π[3], Uniform)
            @test isa(new_continuous.cpt.data.Π[4].ρ, Exponential)
        end
    end
end