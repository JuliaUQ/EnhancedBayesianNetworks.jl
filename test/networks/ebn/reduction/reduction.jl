@testset "Network Reduction" begin

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

    model1 = Model(df -> df.x .^ 2 .- 0.7 .+ df.y, :c1)
    cont_functional1 = ContinuousFunctionalNode(:cf1, [model1], MonteCarlo(300))

    model2 = Model(df -> df.z .^ 2 .- 0.7 .+ df.y, :c2)
    cont_functional2 = ContinuousFunctionalNode(:cf2, [model2], MonteCarlo(300))

    model3 = Model(df -> df.c1 .* 0.5 .+ df.c2, :final1)
    performance1 = df -> df.final1 .- 0.5
    discrete_functional1 = DiscreteFunctionalNode(:fd1, [model3], performance1, MonteCarlo(300), Dict(:fd1_fail => [Parameter(1, :fd1)], :fd1_safe => [Parameter(0, :fd1)]))

    model4 = Model(df -> df.c2 .* 0.5, :c3)
    cont_functional3 = ContinuousFunctionalNode(:c3, [model4], MonteCarlo(300))

    model5 = Model(df -> 0.5 .+ df.c3, :tot)
    performance2 = df -> 0.5 .- df.tot
    discrete_functional = DiscreteFunctionalNode(:fd, [model5], performance2, MonteCarlo(300))

    nodes = [root1, root2, root3, cont_functional1, cont_functional2, discrete_functional1, cont_functional3, discrete_functional]

    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, root1, cont_functional1)
    add_child!(ebn, root2, cont_functional1)
    add_child!(ebn, root2, cont_functional2)
    add_child!(ebn, root3, cont_functional2)
    add_child!(ebn, cont_functional1, discrete_functional1)
    add_child!(ebn, cont_functional2, discrete_functional1)
    add_child!(ebn, cont_functional2, cont_functional3)
    add_child!(ebn, discrete_functional1, discrete_functional)
    add_child!(ebn, cont_functional3, discrete_functional)
    order!(ebn)
    net1 = deepcopy(ebn)

    @test isnothing(reduce!(net1))
    @test net1.A == [
        0.0 0.0 1.0 0.0;
        0.0 0.0 1.0 1.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0
    ]
    @test all(isa.(net1.nodes, DiscreteNode))
    @test net1.topology == Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)

    evaluate!(ebn)
    net2 = deepcopy(ebn)
    reduce!(ebn)
    @test ebn.A == net1.A
    @test all(isa.(ebn.nodes, DiscreteNode))
    @test ebn.topology == Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)

    EnhancedBayesianNetworks._eliminate_continuous_node!(net2, root2)

    @test net2.A == [
        0.0 0.0 1.0 0.0;
        0.0 0.0 1.0 1.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0
    ]
    @test length(net2.nodes) == 4
    @test net2.topology == Dict(:fd => 4, :fd1 => 3, :z => 2, :x => 1)
end