@testset "Evaluation Net" begin

    @testset "Main Functions" begin
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

        evaluate!(ebn)

        @test ebn.A == sparse([0.0 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 0.0 1.0; 0.0 0.0 0.0 0.0 0.0])
        @test ebn.topology == Dict(:y => 2, :fd => 5, :fd1 => 4, :z => 3, :x => 1)
        @test isa(ebn.nodes[4], DiscreteNode)
        @test isroot(ebn.nodes[4]) == false
        @test isa(ebn.nodes[5], DiscreteNode)
        @test isroot(ebn.nodes[5]) == false

        evaluate!(net1, true, false)
        @test isempty(net1.nodes[end].additional_info)

        interval = (1.10, 1.30)
        cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
        cpt_root1[:A=>:a1] = 0.5
        cpt_root1[:A=>:a2] = 0.5
        root1 = DiscreteNode(:A, cpt_root1, Dict(:a1 => [Parameter(1, :A)], :a2 => [Parameter(2, :A)]))

        cpt_root2 = ContinuousConditionalProbabilityTable{Tuple{Real,Real}}()
        cpt_root2[] = interval
        root2 = ContinuousNode(:B, cpt_root2)

        cpt_root3 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root3[] = Uniform(-10, 10)
        root3 = ContinuousNode(:P, cpt_root3)

        model = Model(df -> df.A .+ df.B .+ df.P, :C)
        sim = DoubleLoop(MonteCarlo(100_000))
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)

        nodes = [root1, root2, root3, disc_functional]
        ebn = EnhancedBayesianNetwork(nodes)
        add_child!(ebn, root1, disc_functional)
        add_child!(ebn, root2, disc_functional)
        add_child!(ebn, root3, disc_functional)
        order!(ebn)

        evaluate!(ebn)

        @test isprecise(ebn.nodes[end]) == false
    end

    @testset "Evaluate with envelopes" begin
        n = 10^6
        using .MathConstants: γ

        cpt_Uᵣ = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_Uᵣ[] = Normal()
        Uᵣ = ContinuousNode(:Uᵣ, cpt_Uᵣ)

        cpt_M = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:M)
        cpt_M[:M=>:new] = 0.5
        cpt_M[:M=>:old] = 0.5
        M = DiscreteNode(:M, cpt_M)

        μ_gamma = 60
        cov_gamma = 0.2
        α, θ = distribution_parameters(μ_gamma, μ_gamma * cov_gamma, Gamma)
        cpt_V = ContinuousConditionalProbabilityTable{PreciseContinuousInput}(:M)
        cpt_V[:M=>:new] = Gamma(α, θ)
        cpt_V[:M=>:old] = Gamma(α - 1, 2.4)
        V = ContinuousNode(:V, cpt_V)

        μ_gumbel = 50
        cov_gumbel = 0.4
        μ_loc, β = distribution_parameters(μ_gumbel, cov_gumbel * μ_gumbel, Gumbel)
        cpt_H = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_H[] = Gumbel(μ_loc, β)
        H = ContinuousNode(:H, cpt_H)

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
        R1 = ContinuousFunctionalNode(:R1, [model1], MonteCarlo(n))
        R2 = ContinuousFunctionalNode(:R2, [model2], MonteCarlo(n))
        R3 = ContinuousFunctionalNode(:R3, [model3], MonteCarlo(n))
        R4 = ContinuousFunctionalNode(:R4, [model4], MonteCarlo(n))
        R5 = ContinuousFunctionalNode(:R5, [model5], MonteCarlo(n))
        function frame_model(r1, r2, r3, r4, r5, v, h)
            g1 = r1 + r2 + r4 + r5 - 5 * h
            g2 = r2 + 2 * r3 + r4 - 5 * v
            g3 = r1 + 2 * r3 + 2 * r4 + r5 - 5 * h - 5 * v
            return minimum([g1, g2, g3])
        end
        model = Model(df -> frame_model.(df.r1, df.r2, df.r3, df.r4, df.r5, df.V, df.H), :G)
        performance = df -> df.G
        simulation = MonteCarlo(n)

        frame = DiscreteFunctionalNode(:E, [model], performance, simulation)

        L = DiscreteNode(:L, DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(DataFrame(:M => [:new, :new, :old, :old], :L => [:yesL, :noL, :yesL, :noL], :Π => [0.2, 0.8, 0.5, 0.5])), Dict(:noL => [Parameter(1, :L)], :yesL => [Parameter(2, :L)]))

        cpt_r9 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_r9[] = Normal()
        r9 = ContinuousNode(:R9, cpt_r9)

        model2 = Model(df -> df.L .^ 2 .* df.R9, :P)
        frame2 = DiscreteFunctionalNode(:E2, [model2], df -> df.P, simulation)

        nodes = [Uᵣ, M, V, H, R1, R2, R3, R4, R5, r9, frame, L, frame2]

        ebn = EnhancedBayesianNetwork(nodes)
        add_child!(ebn, M, V)
        add_child!(ebn, Uᵣ, R1)
        add_child!(ebn, Uᵣ, R2)
        add_child!(ebn, Uᵣ, R3)
        add_child!(ebn, Uᵣ, R4)
        add_child!(ebn, Uᵣ, R5)
        add_child!(ebn, R1, frame)
        add_child!(ebn, R2, frame)
        add_child!(ebn, R3, frame)
        add_child!(ebn, R4, frame)
        add_child!(ebn, R5, frame)
        add_child!(ebn, V, frame)
        add_child!(ebn, H, frame)
        add_child!(ebn, M, L)
        add_child!(ebn, L, frame2)
        add_child!(ebn, r9, frame2)
        order!(ebn)
        envelopes = markov_envelope(ebn)

        @test issetequal(EnhancedBayesianNetworks._add_root2envelope(ebn, envelopes[2]), [L, frame2, r9, M])
        @test issetequal(EnhancedBayesianNetworks._add_root2envelope(ebn, envelopes[1]), envelopes[1])

        ebn1 = EnhancedBayesianNetworks._build_envelope_edges(ebn, envelopes[1])
        ebn2 = EnhancedBayesianNetworks._build_envelope_edges(ebn, envelopes[2])

        @test ebn1.A == sparse([
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])

        @test ebn1.topology == Dict(:M => 2, :H => 1, :R2 => 6, :R5 => 9, :V => 4, :R4 => 8, :R3 => 7, :R1 => 5, :E => 10, :Uᵣ => 3)
        @test issetequal(ebn1.nodes, [Uᵣ, M, V, H, R1, R2, R3, R4, R5, frame])

        @test ebn2.A == sparse([
            0.0 0.0 0.0 1.0;
            0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 1.0;
            0.0 0.0 0.0 0.0])
        @test ebn2.topology == Dict(:R9 => 1, :M => 2, :L => 3, :E2 => 4)
        @test issetequal(ebn2.nodes, [r9, L, frame2, M])

        eebns = evaluate_with_envelopes(ebn)
        @test any(isa.(eebns[1].nodes, FunctionalNode)) == false
        @test any(isa.(eebns[2].nodes, FunctionalNode)) == false
    end

    @testset "No Ancestors case" begin
        cpt_root2 = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
        cpt_root2[] = Normal()
        root2 = ContinuousNode(:B, cpt_root2)

        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C
        cont_functional = ContinuousFunctionalNode(:C, [model], sim)

        ebn = EnhancedBayesianNetwork([root2, cont_functional])
        add_child!(ebn, root2, cont_functional)
        order!(ebn)
        evaluate!(ebn)

        @test isa(ebn.nodes[end], ContinuousNode)
        @test isroot(ebn.nodes[end])

        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root2, disc_functional])
        add_child!(ebn, root2, disc_functional)
        order!(ebn)
        evaluate!(ebn)

        @test isa(ebn.nodes[end], DiscreteNode)
        @test isroot(ebn.nodes[end])
    end

    @testset "Imprecise Node with discretization" begin
        cpt_root1 = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:A)
        cpt_root1[:A=>:y] = 0.5
        cpt_root1[:A=>:n] = 0.5
        root1 = DiscreteNode(:A, cpt_root1)

        cpt_root2 = ContinuousConditionalProbabilityTable{UnamedProbabilityBox}(:A)
        cpt_root2[:A=>:y] = UnamedProbabilityBox{Normal}([Interval(-0.5, 0.5, :μ), Interval(1, 2, :σ)])
        cpt_root2[:A=>:n] = UnamedProbabilityBox{Normal}([Interval(1, 2, :μ), Interval(1, 2, :σ)])
        root2 = ContinuousNode(:B, cpt_root2, ApproximatedDiscretization([-1, 0, 1], 2))

        model = Model(df -> df.B .+ 1, :C)
        sim = DoubleLoop(MonteCarlo(100_000))
        performance = df -> 2 .- df.C
        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root1, root2, disc_functional])
        add_child!(ebn, root1, root2)
        add_child!(ebn, root2, disc_functional)
        order!(ebn)

        @test_throws ErrorException("node C has as simulation DoubleLoop(MonteCarlo(100000)), but its imprecise parents will be discretized and approximated with Uniform and Exponential assumption, therefore are no longer imprecise. A prices simulation technique must be selected!") @suppress evaluate!(ebn)

        cpt_root2 = ContinuousConditionalProbabilityTable{Tuple{Real,Real}}()
        cpt_root2[] = (-1, 1)
        root2 = ContinuousNode(:B, cpt_root2)
        model = Model(df -> df.B .+ 1, :C)
        sim = MonteCarlo(100_000)
        performance = df -> 2 .- df.C

        disc_functional = DiscreteFunctionalNode(:C, [model], performance, sim)
        ebn = EnhancedBayesianNetwork([root2, disc_functional])
        add_child!(ebn, root2, disc_functional)
        order!(ebn)

        @test_throws ErrorException("node C has MonteCarlo(100000) as simulation technique, but have [:B] as imprecise parent/s. DoubleLoop or RandomSlicing technique must be employeed instead") @suppress evaluate!(ebn)
    end

    @testset "Auxiliary functions" begin
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
        discrete_functional1 = DiscreteFunctionalNode(:fd1, [model3], performance1, MonteCarlo(300), Dict(:fail_fd1 => [Parameter(1, :fd1)], :safe_fd1 => [Parameter(0, :fd1)]))

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

        @test_throws ErrorException("node elimination algorithm is for continuous nodes and x is discrete") EnhancedBayesianNetworks._is_eliminable(ebn, root1)

        @test EnhancedBayesianNetworks._is_eliminable(ebn, cont_functional2) == false
        @test EnhancedBayesianNetworks._is_eliminable(ebn, root2)
        @test EnhancedBayesianNetworks._is_eliminable(ebn, 2)
        @test EnhancedBayesianNetworks._is_eliminable(ebn, :y)
    end
end