@testset "Bayesian Networks" begin

    cpt_r = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
    cpt_r[] = Normal()
    r = ContinuousNode(:R, cpt_r)

    cpt_v = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:V)
    cpt_v[:V=>:yesV] = 0.01
    cpt_v[:V=>:noV] = 0.99
    v = DiscreteNode(:V, cpt_v, Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))

    cpt_s = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:S)
    cpt_s[:S=>:yesS] = 0.5
    cpt_s[:S=>:noS] = 0.5
    s = DiscreteNode(:S, cpt_s)

    cpt_t = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:V, :T])
    cpt_t[:V=>:yesV, :T=>:yesT] = 0.05
    cpt_t[:V=>:yesV, :T=>:noT] = 0.95
    cpt_t[:V=>:noV, :T=>:yesT] = 0.01
    cpt_t[:V=>:noV, :T=>:noT] = 0.99
    t = DiscreteNode(:T, cpt_t)

    cpt_l = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:S, :L])
    cpt_l[:S=>:yesS, :L=>:yesL] = 0.1
    cpt_l[:S=>:yesS, :L=>:noL] = 0.9
    cpt_l[:S=>:noS, :L=>:yesL] = 0.01
    cpt_l[:S=>:noS, :L=>:noL] = 0.99
    l = DiscreteNode(:L, cpt_l)

    f1 = DiscreteFunctionalNode(
        :F1, [Model(df -> df.v1 .+ df.R, :f1)], df -> 0.8 .- df.f1, MonteCarlo(200)
    )
    nodes = [r, v, s, t, f1]
    @test_throws ErrorException("node/s [:F1] are functional nodes. evaluate the related EnhancedBayesianNetwork structure before!") BayesianNetwork(nodes)

    nodes = [r, v, s, t]
    @test_throws ErrorException("node/s [:R] are continuous. Use EnhancedBayesianNetwork structure!") BayesianNetwork(nodes)

    cpt_r = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
    cpt_r[] = Normal()
    r = ContinuousNode(:V, cpt_r)
    nodes = [r, v, s, t, f1]
    @test_throws ErrorException("network nodes names must be unique") BayesianNetwork(nodes)

    r = ContinuousNode(:R, cpt_r)

    cpt_v = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:V)
    cpt_v[:V=>:yesT] = 0.01
    cpt_v[:V=>:noT] = 0.99
    v = DiscreteNode(:V, cpt_v, Dict(:yesT => [Parameter(0, :v1)], :noT => [Parameter(1, :v1)]))
    nodes = [r, v, s, t, f1]
    @test_throws ErrorException("network nodes states must be unique") BayesianNetwork(nodes)

    cpt_v_imp = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}(:V)
    cpt_v_imp[:V=>:yesV] = (0.01, 0.3)
    cpt_v_imp[:V=>:noV] = (0.7, 0.99)
    v_imp = DiscreteNode(:V, cpt_v_imp, Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))

    nodes = [v_imp, s, t]
    @test_throws ErrorException("node/s [:V] are imprecise. Use CrealNetwork structure!") BayesianNetwork(nodes)

    cpt_v = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:V)
    cpt_v[:V=>:yesV] = 0.01
    cpt_v[:V=>:noV] = 0.99
    v = DiscreteNode(:V, cpt_v, Dict(:yesV => [Parameter(0, :v1)], :noV => [Parameter(1, :v1)]))
    s = DiscreteNode(:S, cpt_s)
    t = DiscreteNode(:T, cpt_t)
    l = DiscreteNode(:L, cpt_l)

    bn = BayesianNetwork([v, s, t, l])

    add_child!(bn, v, t)
    add_child!(bn, s, l)
    order!(bn)

    @test bn.A == sparse([
        0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0])
    @test bn.topology == Dict(:T => 3, :L => 4, :V => 1, :S => 2)
    @test issetequal(bn.nodes, [v, s, t, l])

    ebn = EnhancedBayesianNetwork([v, s, t, l])
    add_child!(ebn, v, t)
    add_child!(ebn, s, l)
    order!(ebn)
    reduce!(ebn)
    bn2 = BayesianNetwork(ebn)

    @test bn2 == bn

    @testset "joint probability" begin
        weather_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:W)
        weather_cpt[:W=>:Cloudy] = 0.5
        weather_cpt[:W=>:Sunny] = 0.5
        weather = DiscreteNode(:W, weather_cpt)

        rain_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:W, :R])
        rain_cpt[:W=>:Cloudy, :R=>:Yes] = 0.8
        rain_cpt[:W=>:Cloudy, :R=>:No] = 0.2
        rain_cpt[:W=>:Sunny, :R=>:Yes] = 0.1
        rain_cpt[:W=>:Sunny, :R=>:No] = 0.9
        rain = DiscreteNode(:R, rain_cpt)

        sprinkler_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:W, :S])
        sprinkler_cpt[:W=>:Cloudy, :S=>:On] = 0.4
        sprinkler_cpt[:W=>:Cloudy, :S=>:Off] = 0.6
        sprinkler_cpt[:W=>:Sunny, :S=>:On] = 0.7
        sprinkler_cpt[:W=>:Sunny, :S=>:Off] = 0.3
        sprinkler = DiscreteNode(:S, sprinkler_cpt)

        grass_cpt = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:R, :S, :G])
        grass_cpt[:R=>:Yes, :S=>:On, :G=>:Wet] = 0.99
        grass_cpt[:R=>:Yes, :S=>:On, :G=>:Dry] = 0.01
        grass_cpt[:R=>:Yes, :S=>:Off, :G=>:Wet] = 0.9
        grass_cpt[:R=>:Yes, :S=>:Off, :G=>:Dry] = 0.1
        grass_cpt[:R=>:No, :S=>:On, :G=>:Wet] = 0.9
        grass_cpt[:R=>:No, :S=>:On, :G=>:Dry] = 0.1
        grass_cpt[:R=>:No, :S=>:Off, :G=>:Wet] = 0.1
        grass_cpt[:R=>:No, :S=>:Off, :G=>:Dry] = 0.9
        grass = DiscreteNode(:G, grass_cpt)

        nodes = [weather, rain, sprinkler, grass]
        bn = BayesianNetwork(nodes)
        add_child!(bn, :W, :R)
        add_child!(bn, :W, :S)
        add_child!(bn, :R, :G)
        add_child!(bn, :S, :G)
        order!(bn)

        scenario1 = Evidence(:W => :Cloudy, :G => :Wet)
        @test_throws ErrorException("Not all the BN's nodes [:W, :R, :S, :G] have a specidied states in Dict(:G => :Wet, :W => :Cloudy). Use Inference!") joint_probability(bn, scenario1)

        scenario2 = Evidence(:W => :Cloudy, :G => :Mild, :R => :Yes, :S => :On)
        @test_throws ErrorException("node G has a defined scenario state Mild that is not among its possible states [:Dry, :Wet]") joint_probability(bn, scenario2)

        scenario3 = Evidence(:W => :Cloudy, :G => :Wet, :R => :Yes, :S => :On, :N => :nothing)
        @test_logs (:warn, "nodes N is not part of the BN, therefore is useless for the scenario probability evaluation") joint_probability(bn, scenario3)

        scenario4 = Evidence(:W => :Cloudy, :G => :Wet, :R => :Yes, :S => :On)
        @test isapprox(joint_probability(bn, scenario4), 0.1584)

        scenario5 = Evidence(:W => :Cloudy, :G => :Dry, :R => :Yes, :S => :On)
        @test isapprox(joint_probability(bn, scenario5), 0.0016)
    end
end

