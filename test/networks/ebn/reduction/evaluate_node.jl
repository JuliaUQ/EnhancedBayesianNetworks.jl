@testsnippet SetupEvaluateeBN begin
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
end

@testitem "Simulation Inputs" setup=[SetupEvaluateeBN] begin
    uqi = EnhancedBayesianNetworks._simulation_inputs(net, E, Dict(:A => :a1, :C => :c1))
    @test isa(uqi, Vector{UQInput})
    @test issetequal(uqi, [Parameter(1, :A), Parameter(1, :C), RandomVariable(Normal(), :D)])

end

@testitem "Simulation Scenarios" setup=[SetupEvaluateeBN] begin
    EnhancedBayesianNetworks._build_simulations!(net, E)
    scs = EnhancedBayesianNetworks._simulation_scenarios(E)
    @test isa(scs, Vector{Evidence})
    @test issetequal(scs, [Evidence(:A => :a1, :C => :c1), Evidence(:A => :a2, :C => :c1), Evidence(:A => :a1, :C => :c2), Evidence(:A => :a2, :C => :c2)])
end

@testitem "Evaluate Node - continuous precise" setup=[SetupEvaluateeBN, ExtraDeps] begin
    EnhancedBayesianNetworks._build_simulations!(net, E)
    evaluated_E = @suppress EnhancedBayesianNetworks.evaluate(net, E)
    @test evaluated_E.name == E.name
    @test all(isa.(evaluated_E.cpt.data.Π, EmpiricalDistribution))
    @test isempty(evaluated_E.discretization)
    @test isa(evaluated_E.results, EnhancedBayesianNetworks.ScenariosTable{Any})
    evaluated_E = @suppress EnhancedBayesianNetworks.evaluate(net, E, false)
    @test evaluated_E.name == E.name
    @test all(isa.(evaluated_E.cpt.data.Π, EmpiricalDistribution))
    @test isempty(evaluated_E.discretization)
    @test isnothing(evaluated_E.results)
end

@testitem "Evaluate Node - continuous imprecise" setup=[SetupEvaluateeBN, ExtraDeps] begin
    EnhancedBayesianNetworks._build_simulations!(net, F)
    evaluated_F = @suppress EnhancedBayesianNetworks.evaluate(net, F)
    @test evaluated_F.name == F.name
    @test all(isa.(evaluated_F.cpt.data.Π, Vector{Pair{Symbol,EmpiricalDistribution}}))
    @test isempty(evaluated_F.discretization)
    @test isa(evaluated_F.results, EnhancedBayesianNetworks.ScenariosTable{Any})
    evaluated_F = @suppress EnhancedBayesianNetworks.evaluate(net, F, false)
    @test evaluated_F.name == F.name
    @test all(isa.(evaluated_F.cpt.data.Π, Vector{Pair{Symbol,EmpiricalDistribution}}))
    @test isempty(evaluated_F.discretization)
    @test isnothing(evaluated_F.results)
end

@testitem "Evaluate Node - discrete precise" setup=[SetupEvaluateeBN] begin
    EnhancedBayesianNetworks._build_simulations!(net, G)
    evaluated_G = EnhancedBayesianNetworks.evaluate(net, G)
    @test evaluated_G.name == G.name
    @test all(isa.(evaluated_G.cpt.data.Π, Real))
    @test isempty(evaluated_G.parameters)
    @test isa(evaluated_G.results, EnhancedBayesianNetworks.ScenariosTable{Any})
    evaluated_G = EnhancedBayesianNetworks.evaluate(net, G, false)
    @test evaluated_G.name == G.name
    @test all(isa.(evaluated_G.cpt.data.Π, Real))
    @test isempty(evaluated_G.parameters)
    @test isnothing(evaluated_G.results)
end

@testitem "Evaluate Node - discrete imprecise" setup=[SetupEvaluateeBN] begin
    EnhancedBayesianNetworks._build_simulations!(net, H)
    evaluated_H = EnhancedBayesianNetworks.evaluate(net, H)
    @test evaluated_H.name == H.name
    @test all(isa.(evaluated_H.cpt.data.Π, Interval))
    @test isempty(evaluated_H.parameters)
    @test isa(evaluated_H.results, EnhancedBayesianNetworks.ScenariosTable{Any})
    evaluated_H = EnhancedBayesianNetworks.evaluate(net, H, false)
    @test evaluated_H.name == H.name
    @test all(isa.(evaluated_H.cpt.data.Π, Interval))
    @test isempty(evaluated_H.parameters)
    @test isnothing(evaluated_H.results)
end

@testitem "Evaluate Node - imprecise simulation error" setup=[ExtraDeps] begin
    A = DiscreteNode(:A, [:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]])
    A[:A=>:a1] = 0.5
    A[:A=>:a2] = 0.5

    # imprecise continuous child of A, carrying a discretization
    V = ContinuousNode(:V, [:A], ApproximatedDiscretization([0.0, 1.0, 2.0], 1))
    V[:A=>:a1] = Interval(0.2, 0.8)
    V[:A=>:a2] = Interval(1.2, 1.8)

    model = Model(df -> df.A .+ df.V, :Z)
    performance = df -> 1 .- df.Z

    # discretizing V moves its imprecision into the credal surrogate V_d and leaves a
    # precise residual feeding E, so an imprecise (double-loop) simulation is invalid
    E = DiscreteFunctionalNode(:E, [model], performance, DoubleLoop(MonteCarlo(100)))
    ebn = EnhancedBayesianNetwork([A, V, E])
    add_child!(ebn, A, V)
    add_child!(ebn, [A, V], E)
    order!(ebn)

    @test_throws ErrorException("Invalid simulation for functional node :E: the assigned DoubleLoop is an imprecise (double-loop) simulation, but every input reaching :E is precise. This happens when an imprecise continuous ancestor of :E is discretized: discretization moves the imprecision into the discrete (credal) surrogate node and leaves a precise continuous residual feeding :E. Use a single-loop simulation (e.g. MonteCarlo) for :E; the network stays credal through the discretized node. Alternatively, remove the discretization from the imprecise continuous ancestor so its imprecision reaches :E directly.") @suppress reduce(ebn)

    # the same network with a single-loop simulation reduces to a CredalNetwork
    E = DiscreteFunctionalNode(:E, [model], performance, MonteCarlo(100))
    ebn = EnhancedBayesianNetwork([A, V, E])
    add_child!(ebn, A, V)
    add_child!(ebn, [A, V], E)
    order!(ebn)
    @test isa((@suppress reduce(ebn)), CredalNetwork)
end