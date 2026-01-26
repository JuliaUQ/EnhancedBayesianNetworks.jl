@testset "Variable Elimination" begin
    v = DiscreteNode(:V, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:V => [:yesV, :noV], :Π => [0.01, 0.99])))
    s = DiscreteNode(:S, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:yesS, :noS], :Π => [0.5, 0.5])))
    t = DiscreteNode(:T, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:V => [:yesV, :yesV, :noV, :noV], :T => [:yesT, :noT, :yesT, :noT], :Π =>
        [0.05, 0.95, 0.01, 0.99])))
    l = DiscreteNode(:L, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:noS, :noS, :yesS, :yesS], :L => [:noL, :yesL, :noL, :yesL], :Π => [0.99, 0.01, 0.9, 0.1])))
    b = DiscreteNode(:B, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:S => [:noS, :noS, :yesS, :yesS], :B => [:noB, :yesB, :noB, :yesB], :Π => [0.7, 0.3, 0.4, 0.6])))
    e = DiscreteNode(:E, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:T => [:noT, :noT, :noT, :noT, :yesT, :yesT, :yesT, :yesT], :L => [:noL, :noL, :yesL, :yesL, :noL, :noL, :yesL, :yesL], :E => [:noE, :yesE, :noE, :yesE, :noE, :yesE, :noE, :yesE], :Π => [1, 0, 0, 1, 0, 1, 0, 1])))
    d = DiscreteNode(:D, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:B => [:noB, :noB, :noB, :noB, :yesB, :yesB, :yesB, :yesB], :E => [:noE, :noE, :yesE, :yesE, :noE, :noE, :yesE, :yesE], :D => [:noD, :yesD, :noD, :yesD, :noD, :yesD, :noD, :yesD], :Π => [0.9, 0.1, 0.3, 0.7, 0.2, 0.8, 0.1, 0.9])))
    x = DiscreteNode(:X, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:E => [:noE, :noE, :yesE, :yesE], :X => [:noX, :yesX, :noX, :yesX], :Π => [0.95, 0.05, 0.02, 0.98])))

    bn = BayesianNetwork([v, s, t, l, b, e, d, x])
    add_child!(bn, v, t)
    add_child!(bn, s, l)
    add_child!(bn, s, b)
    add_child!(bn, t, e)
    add_child!(bn, l, e)
    add_child!(bn, b, d)
    add_child!(bn, e, d)
    add_child!(bn, e, x)
    order!(bn)
    evidence = Evidence()
    factors = map(n -> Factor(bn, n.name, evidence), bn.nodes)
    dimensions = map(f -> f.dimensions, factors)

    adj_moral = EnhancedBayesianNetworks._structure_A(dimensions, bn.topology)

    @test adj_moral == sparse([
        0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 1.0 0.0 0.0 0.0;
        1.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0;
        0.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0;
        0.0 1.0 0.0 0.0 0.0 1.0 1.0 0.0;
        0.0 0.0 1.0 1.0 1.0 0.0 1.0 1.0;
        0.0 0.0 0.0 0.0 1.0 1.0 0.0 0.0;
        0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0
    ])
    @test map(x -> EnhancedBayesianNetworks._n_eliminated_edges(dimensions, bn.topology, x), collect(values(bn.topology))) == [3, 2, 3, 1, 2, 3, 1, 5]
    @test map(x -> EnhancedBayesianNetworks._n_added_edges(dimensions, bn.topology, x), collect(values(bn.topology))) == [2, 0, 2, 0, 1, 2, 0, 8]

    listv = EnhancedBayesianNetworks._order_with_minimal_increase_in_complexity(factors, bn.topology)
    @test listv == [(:D, 0.0), (:V, 0.0), (:X, 0.0), (:S, 0.5), (:T, 0.6666666666666666), (:L, 0.6666666666666666), (:B, 0.6666666666666666), (:E, 1.6)]

    @testset "Inference Precise" begin
        inf = PreciseInferenceState(bn, :B, Dict(:X => :yesX))

        res = infer(inf)

        @test res.dimensions == [:B]
        @test isapprox(res.potential, [0.49367384398446135, 0.5063261560155387])
        @test res.states_mapping == Dict(:B => Dict(:yesB => 2, :noB => 1))

        a = DiscreteNode(:a, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:a => [:yesa, :noa], :Π => [1.0, 0.0])))
        b = DiscreteNode(:b, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:b => [:yesb, :nob], :Π => [0.0, 1.0])))
        c = DiscreteNode(:c, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:a => [:yesa, :yesa, :yesa, :yesa, :noa, :noa, :noa, :noa], :b => [:yesb, :yesb, :nob, :nob, :yesb, :yesb, :nob, :nob], :c => [:yesc, :noc, :yesc, :noc, :yesc, :noc, :yesc, :noc], :Π => [0.1, 0.9, 1.0, 0.0, 0.2, 0.8, 0.4, 0.6])))

        bn = BayesianNetwork([a, b, c])
        add_child!(bn, a, c)
        add_child!(bn, b, c)
        order!(bn)

        ϕ = infer(bn, :a)
        @test length(ϕ) == 2
        f = ϕ[:a=>:yesa]::Factor
        @test isapprox(f.potential[1], 1.0, atol=0.02)
        f = ϕ[:a=>:noa]::Factor
        @test isapprox(f.potential[1], 0.0, atol=0.02)

        ϕ = infer(bn, :c)::Factor
        @test isapprox(ϕ[:c=>:yesc].potential[1], 1.0, atol=0.02)
        @test isapprox(ϕ[:c=>:noc].potential[1], 0.0, atol=0.02)

        ϕ = infer(bn, [:b, :c])

        @test size(ϕ) == (2, 2)
        @test isapprox(ϕ[:b=>:yesb, :c=>:yesc].potential[1], 0.0, atol=0.02)
        @test isapprox(ϕ[:b=>:nob, :c=>:yesc].potential[1], 1.0, atol=0.02)
        @test isapprox(ϕ[:b=>:yesb, :c=>:noc].potential[1], 0.0, atol=0.02)
        @test isapprox(ϕ[:b=>:nob, :c=>:noc].potential[1], 0.0, atol=0.02)

        d = DiscreteNode(:D, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:D => [:noD, :yesD], :Π => [0.4, 0.6])))
        i = DiscreteNode(:I, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:I => [:yesI, :noI], :Π => [0.7, 0.3])))
        g = DiscreteNode(:G, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:D => [:yesD, :yesD, :yesD, :yesD, :yesD, :yesD, :noD, :noD, :noD, :noD, :noD, :noD], :I => [:yesI, :yesI, :yesI, :noI, :noI, :noI, :yesI, :yesI, :yesI, :noI, :noI, :noI], :G => [:firstG, :secondG, :thirdG, :firstG, :secondG, :thirdG, :firstG, :secondG, :thirdG, :firstG, :secondG, :thirdG], :Π => [0.3, 0.4, 0.3, 0.05, 0.25, 0.7, 0.9, 0.08, 0.02, 0.5, 0.3, 0.2])))
        l = DiscreteNode(:L, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:G => [:firstG, :firstG, :secondG, :secondG, :thirdG, :thirdG], :L => [:yesL, :noL, :yesL, :noL, :yesL, :noL], :Π => [0.1, 0.9, 0.4, 0.6, 0.99, 0.01])))
        s = DiscreteNode(:S, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:I => [:yesI, :yesI, :noI, :noI], :S => [:yesS, :noS, :yesS, :noS], :Π => [0.95, 0.05, 0.2, 0.8])))

        bn = BayesianNetwork([d, i, g, l, s])
        add_child!(bn, d, g)
        add_child!(bn, i, g)
        add_child!(bn, g, l)
        add_child!(bn, i, s)
        order!(bn)

        inf = PreciseInferenceState(bn, [:G], Evidence(:D => :yesD, :I => :yesI))
        ϕ = infer(bn, :G, Evidence(:D => :yesD, :I => :yesI))
        @test isapprox(ϕ.potential[1], 0.3, atol=0.05)
        @test isapprox(ϕ.potential[2], 0.4, atol=0.05)
        @test isapprox(ϕ.potential[3], 0.3, atol=0.05)
    end

    @testset "Inference Imprecise" begin
        F = DiscreteNode(:F, DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(DataFrame(:F => [:Ft, :Ff], :Π => [
            (0.4, 0.5), (0.5, 0.6)])))
        B = DiscreteNode(:B, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:B => [:Bt, :Bf], :Π => [0.5, 0.5])))
        L = DiscreteNode(:L, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:F => [:Ft, :Ft, :Ft, :Ff, :Ff, :Ff], :L => [:Lt, :Lf, :L2, :Lt, :Lf, :L2], :Π => [0.3, 0.4, 0.3, 0.05, 0.85, 0.1])))
        D = DiscreteNode(:D, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:F => [:Ft, :Ft, :Ft, :Ft, :Ff, :Ff, :Ff, :Ff], :B => [:Bt, :Bt, :Bf, :Bf, :Bt, :Bt, :Bf, :Bf], :D => [:Dt, :Df, :Dt, :Df, :Dt, :Df, :Dt, :Df], :Π => [0.8, 0.2, 0.1, 0.9, 0.1, 0.9, 0.7, 0.3])))
        H = DiscreteNode(:H, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:D => [:Dt, :Dt, :Df, :Df], :H => [:Ht, :Hf, :Ht, :Hf], :Π => [0.6, 0.4, 0.3, 0.7])))
        cn = CredalNetwork([F, B, L, D, H])
        add_child!(cn, F, L)
        add_child!(cn, F, D)
        add_child!(cn, B, D)
        add_child!(cn, D, H)
        order!(cn)

        evidence = Dict(:D => :Dt)
        query = [:L, :F]

        inference_state = ImpreciseInferenceState(cn, query, evidence)
        ϕ = infer(inference_state)
        mat = reshape([[0.0470588, 0.0571429], [0.4, 0.485714], [0.0235294, 0.0285714], [0.128571, 0.158824], [0.171429, 0.211765], [0.128571, 0.158824]], (3, 2))

        @test isequal(ϕ.dimensions, [:L, :F])
        @test isequal(ϕ.states_mapping, Dict(
            :F => Dict(:Ft => 2, :Ff => 1),
            :L => Dict(:Lt => 3, :Lf => 2, :L2 => 1)))
        @test isapprox(ϕ.potential, mat, atol=0.01)

        ϕ1 = infer(cn, [:L, :F], evidence)
        @test isequal(ϕ.dimensions, ϕ1.dimensions)
        @test isequal(ϕ.states_mapping, ϕ1.states_mapping)
        @test isapprox(ϕ.potential, ϕ1.potential)
    end
    @testset "Straub Example" begin
        using .MathConstants: γ

        n = 10^6
        Uᵣ = ContinuousNode(:Uᵣ, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:Π => Normal())))
        μ_gamma = 60
        cov_gamma = 0.2
        α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
        V = ContinuousNode(:V, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:Π => Gamma(α, θ))))

        μ_gumbel = 50
        cov_gumbel = 0.4
        μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
        H = ContinuousNode(:H, ContinuousConditionalProbabilityTable{PreciseContinuousInput}(DataFrame(:Π => Gumbel(μ_loc, β))))

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

        R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
        R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
        R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))

        @testset "No Evidence" begin
            R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
            R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))

            model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
            performance = df -> df.G
            simulation = MonteCarlo(n)
            frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

            nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

            net = EnhancedBayesianNetwork(nodes)

            add_child!(net, Uᵣ, R1)
            add_child!(net, Uᵣ, R2)
            add_child!(net, Uᵣ, R3)
            add_child!(net, Uᵣ, R4)
            add_child!(net, Uᵣ, R5)
            add_child!(net, R1, frame)
            add_child!(net, R2, frame)
            add_child!(net, R3, frame)
            add_child!(net, R4, frame)
            add_child!(net, R5, frame)
            add_child!(net, V, frame)
            add_child!(net, H, frame)
            order!(net)
            evaluate!(net)

            # @test all(isapprox.(net.nodes[end].cpt[!, :Π], [0.026129, 0.973871]; atol=0.01))
        end

        @testset "Evidence" begin
            n2 = 2000
            discretization1 = ApproximatedDiscretization(collect(range(50, 250, 21)), 1)
            discretization2 = ApproximatedDiscretization(collect(range(50.01, 250.01, 21)), 1)
            R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n2), discretization1)
            R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n2), discretization2)

            model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.R4, df.R5, df.V, df.H), :G)
            performance = df -> df.G
            simulation = MonteCarlo(n2)
            frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

            nodes = [Uᵣ, V, H, R1, R2, R3, R4, R5, frame]

            net = EnhancedBayesianNetwork(nodes)

            add_child!(net, Uᵣ, R1)
            add_child!(net, Uᵣ, R2)
            add_child!(net, Uᵣ, R3)
            add_child!(net, Uᵣ, R4)
            add_child!(net, Uᵣ, R5)
            add_child!(net, R1, frame)
            add_child!(net, R2, frame)
            add_child!(net, R3, frame)
            add_child!(net, R4, frame)
            add_child!(net, R5, frame)
            add_child!(net, V, frame)
            add_child!(net, H, frame)
            order!(net)

            @suppress evaluate!(net)

            @test net.A == sparse([
                0.0 0.0 0.0 1.0 1.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
                0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
            ])
            @test net.topology == Dict(:R5_d => 5,
                :R4_d => 4,
                :H => 3,
                :R5 => 7,
                :V => 2,
                :R4 => 6,
                :E => 8,
                :Uᵣ => 1)

            EnhancedBayesianNetworks._eliminate_continuous_node!(net, net.nodes[7])
            @test net.A == sparse([
                0.0 0.0 0.0 1.0 1.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 1.0 0.0;
                0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 1.0;
                0.0 0.0 0.0 0.0 0.0 0.0 0.0])

            reduce!(net)
            @test net.A == sparse([
                0.0 0.0 1.0;
                0.0 0.0 1.0;
                0.0 0.0 0.0])

            bn = BayesianNetwork(net)

            evidence2 = Dict(
                :R4_d => Symbol([140.0, 150.0]),
                :R5_d => Symbol([90.01, 100.01])
            )
            ϕ2 = infer(bn, :E, evidence2)

            @test all(isapprox.(ϕ2.potential, [0.035, 0.965], atol=0.05))
        end
    end
end