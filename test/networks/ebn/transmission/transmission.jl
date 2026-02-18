@testset "Trasmission Nodes" begin
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
    @suppress order!(net)

    new_discrete_functional = DiscreteFunctionalNode(:fd, [model1, model2], performance, MonteCarlo(300))

    EnhancedBayesianNetworks.transfer_continuous_functional_node!(net, cont_functional)
    adj = sparse([1, 2], [3, 3], [true, true], 3, 3)
    @test net.A == adj
    @test issetequal([i.name for i in net.nodes], [:x, :y, :fd])
    @test net.topology == Dict(:x => 1, :y => 2, :fd => 3)
    @test discrete_functional.models == new_discrete_functional.models

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
    EnhancedBayesianNetworks.transfer_continuous_functional_node!(net, cont_functional)
    @test :fc ∈ [i.name for i in net.nodes]
    @test length(discrete_functional.models) == 1
end