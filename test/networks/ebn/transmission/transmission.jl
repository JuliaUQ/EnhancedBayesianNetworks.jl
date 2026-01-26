@testset "Trasmission Nodes" begin

    @testset "Transfer Single Node" begin
        cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
        cpt_root1[:x=>:x1] = 0.3
        cpt_root1[:x=>:x2] = 0.7
        root1 = DiscreteNode(:x, cpt_root1, Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))

        cpt_root2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root2[] = Normal()
        root2 = ContinuousNode(:y, cpt_root2)

        model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :fc)
        discretization = ApproximatedDiscretization([-2, 0, 2], 2)
        cont_functional = ContinuousFunctionalNode(:fc, [model1], MonteCarlo(300))

        model2 = Model(df -> df.fc .* 0.5, :fd)
        performance = df -> df.fc .- 0.5
        discrete_functional = DiscreteFunctionalNode(:fd, [model2], performance, MonteCarlo(300))

        nodes = [root1, root2, cont_functional, discrete_functional]

        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root1, cont_functional)
        add_child!(net, root2, cont_functional)
        add_child!(net, cont_functional, discrete_functional)
        order!(net)

        new_discrete_functional = DiscreteFunctionalNode(:fd, [model1, model2], performance, MonteCarlo(300))
        EnhancedBayesianNetworks._transfer_single_continuous_functional!(net, cont_functional)

        @test net.A == sparse([0 0 1.0; 0 0 1.0; 0 0 0])
        @test net.topology == Dict(:y => 2, :fd => 3, :x => 1)
        @test issetequal(net.nodes, [root1, root2, new_discrete_functional])

        discretization = ApproximatedDiscretization([-2, 0, 2], 2)
        cont_functional = ContinuousFunctionalNode(:fc, [model1], MonteCarlo(300), discretization)

        nodes = [root1, root2, cont_functional, discrete_functional]

        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root1, cont_functional)
        add_child!(net, root2, cont_functional)
        add_child!(net, cont_functional, discrete_functional)
        order!(net)
        net1 = deepcopy(net)

        EnhancedBayesianNetworks._transfer_single_continuous_functional!(net1, cont_functional)

        @test net1 == net

        cont_functional = ContinuousFunctionalNode(:fc, [model1], MonteCarlo(100))
        discrete_functional = DiscreteFunctionalNode(:fd, [model2], performance, SubSetSimulation(10, 0.1, 10, Uniform(-0.2, 0.2)))

        nodes = [root1, root2, cont_functional, discrete_functional]

        net = EnhancedBayesianNetwork(nodes)
        add_child!(net, root1, cont_functional)
        add_child!(net, root2, cont_functional)
        add_child!(net, cont_functional, discrete_functional)
        order!(net)

        @test_throws ErrorException("node fc cannot be transferred into his children => [:fd], because its simulation type: MonteCarlo is non coherent with children simulation types: DataType[SubSetSimulation]") EnhancedBayesianNetworks._transfer_single_continuous_functional!(net, cont_functional)
    end

    @testset "Transfer All" begin
        cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:x)
        cpt_root1[:x=>:x1] = 0.3
        cpt_root1[:x=>:x2] = 0.7
        root1 = DiscreteNode(:x, cpt_root1, Dict(:x1 => [Parameter(0.5, :x)], :x2 => [Parameter(0.7, :x)]))

        cpt_root2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root2[] = Normal()
        root2 = ContinuousNode(:y, cpt_root2)

        cpt_root3 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:z)
        cpt_root3[:z=>:z1] = 0.3
        cpt_root3[:z=>:z2] = 0.7
        root3 = DiscreteNode(:z, cpt_root3, Dict(:z1 => [Parameter(0.5, :z)], :z2 => [Parameter(0.7, :z)]))

        model1 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c1)
        cont_functional1 = ContinuousFunctionalNode(:cf1, [model1], MonteCarlo(300))

        model2 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c2)
        cont_functional2 = ContinuousFunctionalNode(:cf2, [model2], MonteCarlo(300))

        model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
        performance1 = df -> df.final1 .- 0.5
        discrete_functional1 = DiscreteFunctionalNode(:fd1, [model3], performance1, MonteCarlo(300))

        model4 = Model(df -> df.c2 .* 0.5, :c3)
        cont_functional3 = ContinuousFunctionalNode(:c3, [model4], MonteCarlo(300))

        model5 = Model(df -> df.final2 .* 0.5 .+ df.final1, :tot)
        performance2 = df -> 0.5 .- df.tot
        discrete_functional = DiscreteFunctionalNode(:fd, [model5], performance2, MonteCarlo(300))

        nodes = [root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, cont_functional3, discrete_functional]
        net = EnhancedBayesianNetwork(nodes)

        add_child!(net, root1, cont_functional1)
        add_child!(net, root2, cont_functional1)
        add_child!(net, root2, cont_functional2)
        add_child!(net, root3, cont_functional2)
        add_child!(net, cont_functional1, discrete_functional1)
        add_child!(net, cont_functional2, discrete_functional1)
        add_child!(net, cont_functional2, cont_functional3)
        add_child!(net, discrete_functional1, discrete_functional)
        add_child!(net, cont_functional3, discrete_functional)

        order!(net)

        EnhancedBayesianNetworks._transfer_continuous!(net)

        new_discrete_functional1_1 = DiscreteFunctionalNode(:fd1, [model2, model1, model3], performance1, MonteCarlo(300))
        new_discrete_functional1_2 = DiscreteFunctionalNode(:fd1, [model1, model1, model3], performance1, MonteCarlo(300))

        new_discrete_functional = DiscreteFunctionalNode(:fd, [model2, model4, model5], performance2, MonteCarlo(300))

        @test net.nodes[4] ∈ [new_discrete_functional1_1, new_discrete_functional1_2]
        @test net.nodes[5] == new_discrete_functional
        @test net.A == sparse([0 0 0 1.0 0; 0 0 0 1.0 1.0; 0 0 0 1.0 1.0; 0 0 0 0 1.0; 0 0 0 0 0])
        @test net.topology == Dict(:y => 2, :fd => 5, :fd1 => 4, :z => 3, :x => 1)
    end
end