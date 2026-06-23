@testsnippet NodesSetup begin
    x1 = ContinuousNode(:x1)
    x1[] = Normal()
    x2 = DiscreteNode(:x2)
    x2[:x2=>:yx2] = 0.5
    x2[:x2=>:nx2] = 0.5

    name = :functional
    models = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
    simulation = MonteCarlo(200)

    x1 = ContinuousNode(:x1)
    x1[] = Normal()
    x2 = DiscreteNode(:x2)
    x2[:x2=>:yx2] = 0.5
    x2[:x2=>:nx2] = 0.5

    name = :functional
    models = Model(df -> sqrt.(df.z .^ 2 + df.z .^ 2), :value1)
    simulation = MonteCarlo(200)

    performance = df -> 1 .- 2 .* df.value1

    ancestors = [:x1, :x2, :y1]
    discretization = ApproximatedDiscretization([-2, -1, 0, 1, 2], 2)
end


@testitem "FunctionalNode - Continuous" setup = [NodesSetup] begin
    node = ContinuousFunctionalNode(name, models, simulation)
    @test isa(node, EnhancedBayesianNetworks.AbstractNode)
    @test isa(node, EnhancedBayesianNetworks.AbstractContinuousNode)
    @test isa(node, ContinuousFunctionalNode)
    @test !isa(node, ContinuousNode)
    @test node.name == name
    @test node.models == [models]
    @test node.simulation == simulation
    @test isa(node.discretization, ApproximatedDiscretization)
    @test node.discretization.intervals == Real[]
    @test node.discretization.sigma == 0
    @test node.nbins == 0

    node = ContinuousFunctionalNode(name, models, simulation, discretization)
    @test isa(node.discretization, ApproximatedDiscretization)
    @test node.discretization.intervals == [-2, -1, 0, 1, 2]
    @test node.discretization.sigma == 2
    @test node.nbins == 0
    @test_throws ErrorException(":Π is not allowed as node name") ContinuousFunctionalNode(:Π, models, simulation)
    @test_throws ErrorException(":sim is not allowed as node name") ContinuousFunctionalNode(:sim, models, simulation)

    @test_throws MethodError ContinuousFunctionalNode(name, models, DoubleLoop(MonteCarlo(100)))
    @test_throws MethodError ContinuousFunctionalNode(name, models, RandomSlicing(MonteCarlo(100)))
    @test_throws MethodError ContinuousFunctionalNode(name, models, SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))

    nbins = 100
    node = ContinuousFunctionalNode(name, models, simulation, discretization, nbins)
    @test isa(node.discretization, ApproximatedDiscretization)
    @test node.discretization.intervals == [-2, -1, 0, 1, 2]
    @test node.discretization.sigma == 2
    @test node.nbins == nbins

    node = ContinuousFunctionalNode(name, models, simulation, nbins)
    @test isa(node.discretization, ApproximatedDiscretization)
    @test isempty(node.discretization.intervals)
    @test node.nbins == nbins

    node = ContinuousFunctionalNode(name, ancestors, models)
    node[:x1=>:x1y, :x2=>:x2y, :y1=>:y1y] = MonteCarlo(100)
    node[:x1=>:x1y, :x2=>:x2y, :y1=>:y1n] = MonteCarlo(100)
    node[:x1=>:x1y, :x2=>:x2n, :y1=>:y1y] = MonteCarlo(10)
    node[:x1=>:x1y, :x2=>:x2n, :y1=>:y1n] = MonteCarlo(10)
    node[:x1=>:x1n, :x2=>:x2y, :y1=>:y1y] = MonteCarlo(20)
    node[:x1=>:x1n, :x2=>:x2y, :y1=>:y1n] = MonteCarlo(20)
    node[:x1=>:x1n, :x2=>:x2n, :y1=>:y1y] = MonteCarlo(200)
    node[:x1=>:x1n, :x2=>:x2n, :y1=>:y1n] = MonteCarlo(200)
    @test eltype(node.simulation.data.sim) == AbstractMonteCarlo
    @test node.simulation[:x1=>:x1y, :x2=>:x2y, :y1=>:y1y] == MonteCarlo(100)
    @test node.simulation[:x1=>:x1y, :x2=>:x2y, :y1=>:y1n] == MonteCarlo(100)
    @test node.simulation[:x1=>:x1y, :x2=>:x2n, :y1=>:y1y] == MonteCarlo(10)
    @test node.simulation[:x1=>:x1y, :x2=>:x2n, :y1=>:y1n] == MonteCarlo(10)
    @test node.simulation[:x1=>:x1n, :x2=>:x2y, :y1=>:y1y] == MonteCarlo(20)
    @test node.simulation[:x1=>:x1n, :x2=>:x2y, :y1=>:y1n] == MonteCarlo(20)
    @test node.simulation[:x1=>:x1n, :x2=>:x2n, :y1=>:y1y] == MonteCarlo(200)
    @test node.simulation[:x1=>:x1n, :x2=>:x2n, :y1=>:y1n] == MonteCarlo(200)
    @test node.nbins == 0

    node = ContinuousFunctionalNode(name, ancestors, models, discretization)
    @test node.discretization == discretization
    @test node.nbins == 0

    node = ContinuousFunctionalNode(name, ancestors, models, discretization, nbins)
    @test node.discretization == discretization
    @test node.nbins == nbins

    node = ContinuousFunctionalNode(name, ancestors, models, nbins)
    @test isempty(node.discretization)
    @test node.nbins == nbins

    @test !isroot(node)
end

@testitem "FunctionalNode - Discrete" begin
    node = DiscreteFunctionalNode(name, models, performance, simulation)
    @test isa(node, EnhancedBayesianNetworks.AbstractNode)
    @test isa(node, EnhancedBayesianNetworks.AbstractDiscreteNode)
    @test isa(node, DiscreteFunctionalNode)
    @test !isa(node, DiscreteNode)
    @test node.name == name
    @test node.models == [models]
    @test node.simulation == simulation
    @test node.performance == performance
    @test node.parameters == Vector{Pair{Symbol,Vector{Parameter}}}()
    parameters = [:fail => [Parameter(1, :a), Parameter(2, :b)], :safe => [Parameter(0, :a), Parameter(0, :b)]]
    node = DiscreteFunctionalNode(name, models, performance, simulation, parameters)
    @test node.parameters == parameters
    @test_throws ErrorException(":Π is not allowed as node name") DiscreteFunctionalNode(:Π, models, performance, simulation)
    @test_throws ErrorException(":sim is not allowed as node name") DiscreteFunctionalNode(:sim, models, performance, simulation)

    node = DiscreteFunctionalNode(name, ancestors, models, performance)
    node[:x1=>:x1y, :x2=>:x2y, :y1=>:y1y] = MonteCarlo(100)
    node[:x1=>:x1y, :x2=>:x2y, :y1=>:y1n] = MonteCarlo(100)
    node[:x1=>:x1y, :x2=>:x2n, :y1=>:y1y] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))
    node[:x1=>:x1y, :x2=>:x2n, :y1=>:y1n] = SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))
    node[:x1=>:x1n, :x2=>:x2y, :y1=>:y1y] = DoubleLoop(MonteCarlo(10))
    node[:x1=>:x1n, :x2=>:x2y, :y1=>:y1n] = DoubleLoop(MonteCarlo(10))
    node[:x1=>:x1n, :x2=>:x2n, :y1=>:y1y] = RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
    node[:x1=>:x1n, :x2=>:x2n, :y1=>:y1n] = RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
    @test eltype(node.simulation.data.sim) == EnhancedBayesianNetworks.DiscreteSimulation
    @test node.simulation[:x1=>:x1y, :x2=>:x2y, :y1=>:y1y] == MonteCarlo(100)
    @test node.simulation[:x1=>:x1y, :x2=>:x2y, :y1=>:y1n] == MonteCarlo(100)
    @test node.simulation[:x1=>:x1y, :x2=>:x2n, :y1=>:y1y] == SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))
    @test node.simulation[:x1=>:x1y, :x2=>:x2n, :y1=>:y1n] == SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2))
    @test node.simulation[:x1=>:x1n, :x2=>:x2y, :y1=>:y1y] == DoubleLoop(MonteCarlo(10))
    @test node.simulation[:x1=>:x1n, :x2=>:x2y, :y1=>:y1n] == DoubleLoop(MonteCarlo(10))
    @test node.simulation[:x1=>:x1n, :x2=>:x2n, :y1=>:y1y] == RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))
    @test node.simulation[:x1=>:x1n, :x2=>:x2n, :y1=>:y1n] == RandomSlicing(SubSetSimulation(100, 0.1, 10, Uniform(-0.2, 0.2)))

    node = DiscreteFunctionalNode(name, ancestors, models, performance, parameters)
    @test node.parameters == parameters
    @test issetequal(states(node), Symbol.([string(name) * "_safe", string(name) * "_failed"]))

    @test !isroot(node)
end
