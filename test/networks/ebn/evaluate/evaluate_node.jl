@testset "Evaluation Node" begin
    parameters_A = [:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]]
    A = DiscreteNode(:A, parameters_A)
    A[:A=>:a1] = 0.5
    A[:A=>:a2] = 0.5

    B = ContinuousNode(:B)
    B[] = Interval(1, 2)

    parameters_C = [:c1 => [Parameter(1, :C)], :c2 => [Parameter(2, :C)]]
    C = DiscreteNode(:C, parameters_C)
    C[:C=>:c1] = 0.5
    C[:C=>:c2] = 0.5

    D = ContinuousNode(:D)
    D[] = Normal()

    model = Model(df -> df.A .* df.D .- df.C, :E)
    sim = MonteCarlo(1_000)
    E = ContinuousFunctionalNode(:E, [model], sim, 100)

    model = Model(df -> df.A .* df.D .- df.C .* df.B, :F)
    sim = MonteCarlo(1_000)
    F = ContinuousFunctionalNode(:F, [model], sim, 100)

    model = Model(df -> df.A .- df.C .+ df.D, :G)
    sim = MonteCarlo(1_000)
    performance = df -> 2 .- df.G
    G = DiscreteFunctionalNode(:G, [model], performance, sim)

    model = Model(df -> df.A .- df.C .+ df.D .* df.B, :H)
    sim = DoubleLoop(MonteCarlo(1_000))
    performance = df -> 2 .- df.H
    H = DiscreteFunctionalNode(:H, [model], performance, sim)

    net = EnhancedBayesianNetwork([A, B, C, D, E, F, G, H])
    add_child!(net, [A, C, D], E)
    add_child!(net, [A, D, C, B], F)
    add_child!(net, [A, C, D], G)
    add_child!(net, [A, D, C, B], H)
    order!(net)

    EnhancedBayesianNetworks.build_simulation_table!(net, E)
    @time evaluated_E = EnhancedBayesianNetworks.evaluate(net, E)

    EnhancedBayesianNetworks.build_simulation_table!(net, F)
    @time evaluated_F = EnhancedBayesianNetworks.evaluate(net, F)

    EnhancedBayesianNetworks.build_simulation_table!(net, G)
    @time evaluated_G = EnhancedBayesianNetworks.evaluate(net, G)

    EnhancedBayesianNetworks.build_simulation_table!(net, H)
    @time evaluated_H = EnhancedBayesianNetworks.evaluate(net, H)
end