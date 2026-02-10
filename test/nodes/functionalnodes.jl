@testset "Functional Nodes" begin
    x1 = ContinuousNode(:x1)
    x1[] = Normal()
    x2 = DiscreteNode(:x2)
    x2[:x2=>:yx2] = 0.5
    x2[:x2=>:nx2] = 0.5

    @testset "Continuous" begin
        name = :functional
        models = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        simulation = MonteCarlo(200)
        node = ContinuousFunctionalNode(name, models, simulation)
        @test node.name == name
        @test node.models == [models]
        @test node.simulation == simulation
        @test isa(node.discretization, ApproximatedDiscretization)
        @test node.discretization.intervals == Real[]
        @test node.discretization.sigma == 0
        discretization = ApproximatedDiscretization([-2, -1, 0, 1, 2], 2)
        node = ContinuousFunctionalNode(name, models, simulation, discretization)
        @test isa(node.discretization, ApproximatedDiscretization)
        @test node.discretization.intervals == [-2, -1, 0, 1, 2]
        @test node.discretization.sigma == 2
        @test_throws ErrorException(":Π is not allowed as node name") ContinuousFunctionalNode(:Π, models, simulation)
        @test EnhancedBayesianNetworks.isa_generalized_continuous(node)
        @test EnhancedBayesianNetworks.isa_generalized_continuous(x1)
        @test !EnhancedBayesianNetworks.isa_generalized_continuous(x2)
        @test !EnhancedBayesianNetworks.isa_generalized_discrete(node)
    end

    @testset "Discrete" begin
        name = :functional
        models = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
        simulation = MonteCarlo(200)
        performance = df -> 1 .- 2 .* df.value1
        node = DiscreteFunctionalNode(name, models, performance, simulation)
        @test node.name == name
        @test node.models == [models]
        @test node.simulation == simulation
        @test node.performance == performance
        @test node.parameters == Dict{Symbol,Vector{Parameter}}()
        parameters = Dict(:fail => [Parameter(1, :a), Parameter(2, :b)], :safe => [Parameter(0, :a), Parameter(0, :b)])
        node = DiscreteFunctionalNode(name, models, performance, simulation, parameters)
        @test node.parameters == parameters
        @test_throws ErrorException(":Π is not allowed as node name") DiscreteFunctionalNode(:Π, models, performance, simulation)
        @test EnhancedBayesianNetworks.isa_generalized_discrete(node)
        @test EnhancedBayesianNetworks.isa_generalized_discrete(x2)
        @test !EnhancedBayesianNetworks.isa_generalized_discrete(x1)
        @test !EnhancedBayesianNetworks.isa_generalized_continuous(node)
    end
end
