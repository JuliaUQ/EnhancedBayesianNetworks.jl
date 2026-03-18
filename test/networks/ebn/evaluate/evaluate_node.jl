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

    @testset "simulation inputs" begin
        uqi = EnhancedBayesianNetworks.simulation_inputs(net, E, Dict(:A => :a1, :C => :c1))
        @test isa(uqi, Vector{UQInput})
        @test issetequal(uqi, [Parameter(1, :A), Parameter(1, :C), RandomVariable(Normal(), :D)])
    end

    @testset "simulation scenarios" begin
        EnhancedBayesianNetworks.build_simulations!(net, E)
        scs = EnhancedBayesianNetworks.simulation_scenarios(E)
        @test isa(scs, Vector{Evidence})
        @test issetequal(scs, [Evidence(:A => :a1, :C => :c1), Evidence(:A => :a2, :C => :c1), Evidence(:A => :a1, :C => :c2), Evidence(:A => :a2, :C => :c2)])
    end

    @testset "Continuous Precise" begin
        evaluated_E = EnhancedBayesianNetworks.evaluate(net, E)
        @test evaluated_E.name == E.name
        @test all(isa.(evaluated_E.cpt.data.Π, EmpiricalDistribution))
        @test isempty(evaluated_E.discretization)
        @test isa(evaluated_E.results, EnhancedBayesianNetworks.ScenariosTable{Any})
        evaluated_E = EnhancedBayesianNetworks.evaluate(net, E, false)
        @test evaluated_E.name == E.name
        @test all(isa.(evaluated_E.cpt.data.Π, EmpiricalDistribution))
        @test isempty(evaluated_E.discretization)
        @test isnothing(evaluated_E.results)
    end

    @testset "Continuous Imprecise" begin
        EnhancedBayesianNetworks.build_simulations!(net, F)
        evaluated_F = EnhancedBayesianNetworks.evaluate(net, F)
        @test evaluated_F.name == F.name
        @test all(isa.(evaluated_F.cpt.data.Π, Vector{Pair{Symbol,EmpiricalDistribution}}))
        @test isempty(evaluated_F.discretization)
        @test isa(evaluated_F.results, EnhancedBayesianNetworks.ScenariosTable{Any})
        evaluated_F = EnhancedBayesianNetworks.evaluate(net, F, false)
        @test evaluated_F.name == F.name
        @test all(isa.(evaluated_F.cpt.data.Π, Vector{Pair{Symbol,EmpiricalDistribution}}))
        @test isempty(evaluated_F.discretization)
        @test isnothing(evaluated_F.results)
    end

    @testset "Discrete Precise" begin
        EnhancedBayesianNetworks.build_simulations!(net, G)
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

    @testset "Discrete Imprecise" begin
        EnhancedBayesianNetworks.build_simulations!(net, H)
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
end