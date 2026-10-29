@testsnippet SetupFrameeBN begin
    using .MathConstants: γ

    μ_gamma = 60
    cov_gamma = 0.2
    α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
    V = ContinuousNode(:V, Gamma(α, θ))

    μ_gumbel = 50
    cov_gumbel = 0.4
    μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
    H = ContinuousNode(:H, Gumbel(μ_loc, β))

    function plastic_moment_capacities(uᵣ)
        ρ = 0.5477
        μ = 150
        cov = 0.2
        λ, ζ = distribution_parameters(μ, μ * cov, LogNormal)
        normal_μ = λ + ρ * ζ * uᵣ
        normal_std = sqrt((1 - ρ^2) * ζ^2)
        exp(rand(Normal(normal_μ, normal_std)))
    end

    model1 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r1)
    model2 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r2)
    model3 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r3)
    model4 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r4)
    model5 = Model(df -> plastic_moment_capacities.(df.Uᵣ), :r5)

    function frame_model(r1, r2, r3, r4, r5, v, h)
        g1 = r1 + r2 + r4 + r5 - 5 * h
        g2 = r2 + 2 * r3 + r4 - 5 * v
        g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
        return minimum([g1, g2, g3])
    end

    model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
    performance = df -> df.G

end

@testitem "Evaluate Net - precise no discrete parents" setup = [ExtraDeps, SetupFrameeBN] begin
    n = 10^6
    Uᵣ = ContinuousNode(:Uᵣ, Normal())
    R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
    R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
    R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
    R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
    R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))
    simulation = MonteCarlo(n)
    frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

    nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
    add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
    order!(ebn)

    reduced_ebn = @suppress reduce(ebn)
    @test size(reduced_ebn.A) == (1, 1)
    @test reduced_ebn.topology == Dict(:E => 1)
    @test isa(reduced_ebn.nodes[1], DiscreteNode)
    @test isroot(reduced_ebn.nodes[1])
    @test issetequal(reduced_ebn.nodes[1].cpt.data.E, [:E_failed, :E_safe])
    @test isapprox(reduced_ebn.nodes[1].cpt.data.Π, [0.025693, 0.974307], atol = 0.05)
    @test !isnothing(reduced_ebn.nodes[1].results)

    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
    add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
    order!(ebn)
    reduced_ebn = @suppress reduce(ebn, false)
    @test size(reduced_ebn.A) == (1, 1)
    @test reduced_ebn.topology == Dict(:E => 1)
    @test isa(reduced_ebn.nodes[1], DiscreteNode)
    @test isroot(reduced_ebn.nodes[1])
    @test issetequal(reduced_ebn.nodes[1].cpt.data.E, [:E_failed, :E_safe])
    @test isapprox(reduced_ebn.nodes[1].cpt.data.Π, [0.025693, 0.974307], atol = 0.05)
    @test isnothing(reduced_ebn.nodes[1].results)
end

@testitem "Evaluate Net - imprecise no discrete parents" setup = [ExtraDeps, SetupFrameeBN, CheckSetup] begin
    n = 10^4
    Uᵣ = ContinuousNode(:Uᵣ, Interval(-1, 1))
    R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
    R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
    R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
    R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
    R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))
    simulation = DoubleLoop(MonteCarlo(n))
    frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

    nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
    add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
    order!(ebn)

    ebn1 = deepcopy(ebn)
    ebn1 = @suppress reduce(ebn1)
    @test size(ebn1.A) == (1, 1)
    @test ebn1.topology == Dict(:E => 1)
    @test isa(ebn1.nodes[1], DiscreteNode)
    @test isroot(ebn1.nodes[1])
    @test issetequal(ebn1.nodes[1].cpt.data.E, [:E_failed, :E_safe])
    @test isapprox(ebn1.nodes[1].cpt.data.Π[1].lb, 0, atol = 0.05)
    @test isapprox(ebn1.nodes[1].cpt.data.Π[1].ub, 0.057, atol = 0.05)
    @test isapprox(ebn1.nodes[1].cpt.data.Π[2].lb, 0.943, atol = 0.05)
    @test isapprox(ebn1.nodes[1].cpt.data.Π[2].ub, 1, atol = 0.05)
    @test !isnothing(ebn1.nodes[1].results)
    check_index_coherence(ebn1)

    ebn2 = deepcopy(ebn)
    ebn2 = @suppress reduce(ebn2, false)
    @test size(ebn2.A) == (1, 1)
    @test ebn2.topology == Dict(:E => 1)
    @test isa(ebn2.nodes[1], DiscreteNode)
    @test isroot(ebn2.nodes[1])
    @test issetequal(ebn2.nodes[1].cpt.data.E, [:E_failed, :E_safe])
    @test isapprox(ebn2.nodes[1].cpt.data.Π[1].lb, 0, atol = 0.05)
    @test isapprox(ebn2.nodes[1].cpt.data.Π[1].ub, 0.057, atol = 0.05)
    @test isapprox(ebn2.nodes[1].cpt.data.Π[2].lb, 0.943, atol = 0.05)
    @test isapprox(ebn2.nodes[1].cpt.data.Π[2].ub, 1, atol = 0.05)
    @test isnothing(ebn2.nodes[1].results)
    check_index_coherence(ebn2)
end

@testitem "Evaluate Net - precise discrete parents" setup = [ExtraDeps, SetupFrameeBN] begin

    n = 10^6
    Uᵣ = ContinuousNode(:Uᵣ, Normal())

    M = DiscreteNode(:M)
    M[:M => :new] = 0.5
    M[:M => :old] = 0.5

    μ_gamma = 60
    cov_gamma = 0.2
    α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
    V = ContinuousNode(:V, [:M])
    V[:M => :new] = Gamma(α, θ)
    V[:M => :old] = Gamma(α - 1, 2.4)

    μ_gumbel = 50
    cov_gumbel = 0.4
    μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
    H = ContinuousNode(:H, Gumbel(μ_loc, β))

    R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
    R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
    R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
    R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
    R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))

    simulation = MonteCarlo(n)
    frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

    parameters_L = [:yesL => [Parameter(1, :L)], :noL => [Parameter(2, :L)]]
    L = DiscreteNode(:L, [:M], parameters_L)
    L[:M => :new, :L => :yesL] = 0.2
    L[:M => :new, :L => :noL] = 0.8
    L[:M => :old, :L => :yesL] = 0.5
    L[:M => :old, :L => :noL] = 0.5

    r9 = ContinuousNode(:R9, Normal())

    model2 = Model(df -> df.L .^ 2 .* df.R9, :P)
    frame2 = DiscreteFunctionalNode(:E2, [model2], df -> df.P, simulation)

    nodes = [Uᵣ, M, V, H, R1, R2, R3, R4, R5, r9, frame, L, frame2]
    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, M, [V, L])
    add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
    add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
    add_child!(ebn, [L, r9], frame2)
    order!(ebn)

    ebn1 = deepcopy(ebn)
    ebn1 = @suppress reduce(ebn1)
    @test size(ebn1.A) == (4, 4)
    @test ebn1.topology == Dict(:M => 1, :L => 2, :E => 3, :E2 => 4)
    @test isa(ebn1.nodes[3], DiscreteNode)
    @test !isroot(ebn1.nodes[3])
    @test issetequal(ebn1.nodes[3].cpt.data.E, [:E_failed, :E_safe])
    @test issetequal(ebn1.nodes[3].cpt.data.M, [:new, :old])
    @test isapprox(ebn1.nodes[3].cpt.data.Π[1], 0.025959, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[2], 0.974041, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[3], 0.022452, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[4], 0.977548, atol = 0.05)
    @test !isnothing(ebn1.nodes[3].results)
    @test isa(ebn1.nodes[4], DiscreteNode)
    @test !isroot(ebn1.nodes[4])
    @test issetequal(ebn1.nodes[4].cpt.data.E2, [:E2_failed, :E2_safe])
    @test issetequal(ebn1.nodes[4].cpt.data.L, [:yesL, :noL])
    @test isapprox(ebn1.nodes[4].cpt.data.Π[1], 0.501359, atol = 0.05)
    @test isapprox(ebn1.nodes[4].cpt.data.Π[2], 0.498641, atol = 0.05)
    @test isapprox(ebn1.nodes[4].cpt.data.Π[3], 0.500866, atol = 0.05)
    @test isapprox(ebn1.nodes[4].cpt.data.Π[4], 0.499134, atol = 0.05)
    @test !isnothing(ebn1.nodes[4].results)

    ebn2 = deepcopy(ebn)
    ebn2 = @suppress reduce(ebn2, false)
    @test size(ebn2.A) == (4, 4)
    @test ebn2.topology == Dict(:M => 1, :L => 2, :E => 3, :E2 => 4)
    @test isa(ebn2.nodes[3], DiscreteNode)
    @test !isroot(ebn2.nodes[3])
    @test issetequal(ebn2.nodes[3].cpt.data.E, [:E_failed, :E_safe])
    @test issetequal(ebn2.nodes[3].cpt.data.M, [:new, :old])
    @test isapprox(ebn2.nodes[3].cpt.data.Π[1], 0.025959, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[2], 0.974041, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[3], 0.022452, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[4], 0.977548, atol = 0.05)
    @test isnothing(ebn2.nodes[3].results)
    @test isa(ebn2.nodes[4], DiscreteNode)
    @test !isroot(ebn2.nodes[4])
    @test issetequal(ebn2.nodes[4].cpt.data.E2, [:E2_failed, :E2_safe])
    @test issetequal(ebn2.nodes[4].cpt.data.L, [:yesL, :noL])
    @test isapprox(ebn2.nodes[4].cpt.data.Π[1], 0.501359, atol = 0.05)
    @test isapprox(ebn2.nodes[4].cpt.data.Π[2], 0.498641, atol = 0.05)
    @test isapprox(ebn2.nodes[4].cpt.data.Π[3], 0.500866, atol = 0.05)
    @test isapprox(ebn2.nodes[4].cpt.data.Π[4], 0.499134, atol = 0.05)
    @test isnothing(ebn2.nodes[4].results)
end

@testitem "Evaluate Net - imprecise discrete parents" setup = [ExtraDeps, SetupFrameeBN] begin
    n = 10^4
    Uᵣ = ContinuousNode(:Uᵣ, Normal())

    M = DiscreteNode(:M)
    M[:M => :new] = 0.5
    M[:M => :old] = 0.5

    μ_gamma = 60
    cov_gamma = 0.2
    α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
    V = ContinuousNode(:V, [:M])
    V[:M => :new] = Gamma(α, θ)
    V[:M => :old] = Gamma(α - 1, 2.4)

    μ_gumbel = 50
    cov_gumbel = 0.4
    μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
    dist = ProbabilityBox{Gumbel}(Dict(:μ => Interval(μ_loc - 5, μ_loc + 5), :θ => β))
    H = ContinuousNode(:H, dist)

    R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
    R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
    R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
    R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
    R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))

    simulation = DoubleLoop(MonteCarlo(n))
    frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

    parameters_L = [:yesL => [Parameter(1, :L)], :noL => [Parameter(2, :L)]]
    L = DiscreteNode(:L, [:M], parameters_L)
    L[:M => :new, :L => :yesL] = 0.2
    L[:M => :new, :L => :noL] = 0.8
    L[:M => :old, :L => :yesL] = 0.5
    L[:M => :old, :L => :noL] = 0.5

    r9 = ContinuousNode(:R9, Normal())

    model2 = Model(df -> df.L .^ 2 .* df.R9, :P)
    frame2 = DiscreteFunctionalNode(:E2, [model2], df -> df.P, MonteCarlo(n))

    nodes = [Uᵣ, M, V, H, R1, R2, R3, R4, R5, r9, frame, L, frame2]
    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, M, [V, L])
    add_child!(ebn, Uᵣ, [R1, R2, R3, R4, R5])
    add_child!(ebn, [R1, R2, R3, R4, R5, V, H], frame)
    add_child!(ebn, [L, r9], frame2)
    order!(ebn)

    ebn1 = deepcopy(ebn)
    ebn1 = @suppress reduce(ebn1)
    @test size(ebn1.A) == (4, 4)
    @test ebn1.topology == Dict(:M => 1, :L => 2, :E => 3, :E2 => 4)
    @test isa(ebn1.nodes[3], DiscreteNode)
    @test !isroot(ebn1.nodes[3])
    @test issetequal(ebn1.nodes[3].cpt.data.E, [:E_failed, :E_safe])
    @test issetequal(ebn1.nodes[3].cpt.data.M, [:new, :old])
    @test isapprox(ebn1.nodes[3].cpt.data.Π[1].lb, 0.0171, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[1].ub, 0.0369, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[2].lb, 0.9631, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[2].ub, 0.9829, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[3].lb, 0.0132, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[3].ub, 0.0333, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[4].lb, 0.9667, atol = 0.05)
    @test isapprox(ebn1.nodes[3].cpt.data.Π[4].ub, 0.9868, atol = 0.05)
    @test !isnothing(ebn1.nodes[3].results)
    @test isa(ebn1.nodes[4], DiscreteNode)
    @test !isroot(ebn1.nodes[4])
    @test issetequal(ebn1.nodes[4].cpt.data.E2, [:E2_failed, :E2_safe])
    @test issetequal(ebn1.nodes[4].cpt.data.L, [:yesL, :noL])
    @test isapprox(ebn1.nodes[4].cpt.data.Π[1], 0.501359, atol = 0.05)
    @test isapprox(ebn1.nodes[4].cpt.data.Π[2], 0.498641, atol = 0.05)
    @test isapprox(ebn1.nodes[4].cpt.data.Π[3], 0.500866, atol = 0.05)
    @test isapprox(ebn1.nodes[4].cpt.data.Π[4], 0.499134, atol = 0.05)
    @test !isnothing(ebn1.nodes[4].results)

    ebn2 = deepcopy(ebn)
    ebn2 = @suppress reduce(ebn2, false)
    @test size(ebn2.A) == (4, 4)
    @test ebn2.topology == Dict(:M => 1, :L => 2, :E => 3, :E2 => 4)
    @test isa(ebn2.nodes[3], DiscreteNode)
    @test !isroot(ebn2.nodes[3])
    @test issetequal(ebn2.nodes[3].cpt.data.E, [:E_failed, :E_safe])
    @test issetequal(ebn2.nodes[3].cpt.data.M, [:new, :old])
    @test isapprox(ebn2.nodes[3].cpt.data.Π[1].lb, 0.0171, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[1].ub, 0.0369, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[2].lb, 0.9631, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[2].ub, 0.9829, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[3].lb, 0.0132, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[3].ub, 0.0333, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[4].lb, 0.9667, atol = 0.05)
    @test isapprox(ebn2.nodes[3].cpt.data.Π[4].ub, 0.9868, atol = 0.05)
    @test isnothing(ebn2.nodes[3].results)
    @test isa(ebn2.nodes[4], DiscreteNode)
    @test !isroot(ebn2.nodes[4])
    @test issetequal(ebn2.nodes[4].cpt.data.E2, [:E2_failed, :E2_safe])
    @test issetequal(ebn2.nodes[4].cpt.data.L, [:yesL, :noL])
    @test isapprox(ebn2.nodes[4].cpt.data.Π[1], 0.501359, atol = 0.05)
    @test isapprox(ebn2.nodes[4].cpt.data.Π[2], 0.498641, atol = 0.05)
    @test isapprox(ebn2.nodes[4].cpt.data.Π[3], 0.500866, atol = 0.05)
    @test isapprox(ebn2.nodes[4].cpt.data.Π[4], 0.499134, atol = 0.05)
    @test isnothing(ebn2.nodes[4].results)
end

@testitem "Evaluate Net - degenerate p-box reduces precisely" setup = [ExtraDeps] begin
    # A ProbabilityBox whose every interval parameter has zero width is a precise distribution, and is stored as one
    degenerate = ProbabilityBox{Normal}(Dict(:μ => Interval(0.2, 0.2), :σ => 1))
    X = ContinuousNode(:X, degenerate, ExactDiscretization([-1.0, 0.0, 1.0]))
    model = Model(df -> df.X .+ 1.0, :g)
    F = DiscreteFunctionalNode(:F, [model], df -> df.g, MonteCarlo(200))

    ebn = EnhancedBayesianNetwork([X, F])
    add_child!(ebn, :X, :F)
    order!(ebn)

    reduced = @suppress reduce(ebn)
    @test isa(reduced, BayesianNetwork)

    surrogate = only(filter(n -> n.name == :X_d, reduced.nodes))
    member = Normal(0.2, 1.0)
    expected = [
        cdf(member, -1.0),
        cdf(member, 0.0) - cdf(member, -1.0),
        cdf(member, 1.0) - cdf(member, 0.0),
        1 - cdf(member, 1.0),
    ]
    @test all(isa.(surrogate.cpt.data.Π, Real))
    @test all(isapprox.(surrogate.cpt.data.Π, expected; atol = 1.0e-12))

    X = ContinuousNode(:X, degenerate, ExactDiscretization([-1.0, 0.0, 1.0]))
    F = DiscreteFunctionalNode(:F, [model], df -> df.g, DoubleLoop(MonteCarlo(200)))
    ebn = EnhancedBayesianNetwork([X, F])
    add_child!(ebn, :X, :F)
    order!(ebn)
    @test_throws "is an imprecise (double-loop) simulation, but every input reaching :F is precise" @suppress reduce(ebn)
end

@testitem "Evaluate Net - degenerate interval probabilities reduce precisely" setup = [ExtraDeps] begin
    # Zero-width interval probabilities describe a precise CPT, so the network is a BayesianNetwork and inference stays exact.
    A = DiscreteNode(:A)
    A[:A => :a1] = Interval(0.3, 0.3)
    A[:A => :a2] = Interval(0.7, 0.7)
    B = DiscreteNode(:B, [:A])
    B[:A => :a1, :B => :b1] = 0.9
    B[:A => :a1, :B => :b2] = 0.1
    B[:A => :a2, :B => :b1] = 0.2
    B[:A => :a2, :B => :b2] = 0.8

    net = EnhancedBayesianNetwork([A, B])
    add_child!(net, :A, :B)
    order!(net)

    reduced = @suppress reduce(net)
    @test isa(reduced, BayesianNetwork)
    root = only(filter(n -> n.name == :A, reduced.nodes))
    @test all(isa.(root.cpt.data.Π, Real))
    @test root.cpt.data.Π == [0.3, 0.7]
end


@testitem "Evaluate Net - Auxiliary Functions" begin
    A = DiscreteNode(:A)
    A[:A => :a1] = 0.5
    A[:A => :a2] = 0.5

    parameters_B = [:b1 => [Parameter(1, :B)], :b2 => [Parameter(0, :B)]]
    B = DiscreteNode(:B, parameters_B)
    B[:B => :b1] = 0.5
    B[:B => :b2] = 0.5

    C = ContinuousNode(:C, [:A])
    C[:A => :a1] = Normal(0, 1)
    C[:A => :a1] = Normal(0.5, 1)

    model_D = Model(df -> df.B .- df.C, :D)
    simulation = MonteCarlo(1000)
    D = ContinuousFunctionalNode(:D, model_D, simulation)

    parameters_E = [:e1 => [Parameter(1, :E)], :e2 => [Parameter(0, :E)]]
    E = DiscreteNode(:E, parameters_E)
    E[:E => :e1] = 0.5
    E[:E => :e2] = 0.5

    model_F = Model(df -> df.D .+ df.E, :F)
    performance = df -> df.F
    F = DiscreteFunctionalNode(:F, model_F, performance, simulation)


    nodes = [A, B, C, D, E, F]
    ebn = EnhancedBayesianNetwork(nodes)
    add_child!(ebn, A, C)
    add_child!(ebn, [B, C], D)
    add_child!(ebn, [D, E], F)
    order!(ebn)

    @test EnhancedBayesianNetworks._has_functional_parents(ebn, F)
    @test !EnhancedBayesianNetworks._has_functional_parents(ebn, D)

    ebn1 = deepcopy(ebn)
    EnhancedBayesianNetworks._eliminate_node!(ebn1, C)
    @test C.name ∉ getproperty.(ebn1.nodes, :name)
end

@testitem "Evaluate Net - progress kwarg" setup = [ExtraDeps] begin
    Load = DiscreteNode(:Load, [:low => [Parameter(1.0, :Load)], :high => [Parameter(3.0, :Load)]])
    Load[:Load => :low] = 0.7
    Load[:Load => :high] = 0.3
    R = ContinuousNode(:R, Normal(3.0, 0.5))
    model = Model(df -> df.R .- df.Load, :g)
    F = DiscreteFunctionalNode(:F, [model], df -> df.g, MonteCarlo(200))

    ebn = EnhancedBayesianNetwork([Load, R, F])
    add_child!(ebn, :Load, :F)
    add_child!(ebn, :R, :F)
    order!(ebn)

    # forcing the progress bar on must not change what reduce returns
    reduced = @suppress reduce(ebn; progress = true)
    @test isa(reduced, BayesianNetwork)
end
