@testset "Functional Nodes" begin
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
    end
end
