@testset "Credal Networks" begin

    cpt_f = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:F)
    cpt_f[:F=>:Ft] = 0.5
    cpt_f[:F=>:Ff] = 0.5
    F = DiscreteNode(:F, cpt_f)

    cpt_b = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}(:B)
    cpt_b[:B=>:Bt] = 0.5
    cpt_b[:B=>:Bf] = 0.5
    B = DiscreteNode(:B, cpt_b)

    cpt_l = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:F, :L])
    cpt_l[:F=>:Ft, :L=>:Lt] = 0.3
    cpt_l[:F=>:Ft, :L=>:Lf] = 0.4
    cpt_l[:F=>:Ft, :L=>:L2] = 0.3
    cpt_l[:F=>:Ff, :L=>:Lt] = 0.05
    cpt_l[:F=>:Ff, :L=>:Lf] = 0.85
    cpt_l[:F=>:Ff, :L=>:L2] = 0.1
    L = DiscreteNode(:L, cpt_l)

    cpt_d = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:F, :B, :D])
    cpt_d[:F=>:Ft, :B=>:Bt, :D=>:Dt] = 0.8
    cpt_d[:F=>:Ft, :B=>:Bt, :D=>:Df] = 0.2
    cpt_d[:F=>:Ft, :B=>:Bf, :D=>:Dt] = 0.1
    cpt_d[:F=>:Ft, :B=>:Bf, :D=>:Df] = 0.9
    cpt_d[:F=>:Ff, :B=>:Bt, :D=>:Dt] = 0.1
    cpt_d[:F=>:Ff, :B=>:Bt, :D=>:Df] = 0.9
    cpt_d[:F=>:Ff, :B=>:Bf, :D=>:Dt] = 0.7
    cpt_d[:F=>:Ff, :B=>:Bf, :D=>:Df] = 0.3
    D = DiscreteNode(:D, cpt_d)

    cpt_h = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:D, :B])
    cpt_h[:D=>:Dt, :B=>:Ht] = 0.6
    cpt_h[:D=>:Dt, :B=>:Hf] = 0.4
    cpt_h[:D=>:Df, :B=>:Ht] = 0.3
    cpt_h[:D=>:Df, :B=>:Hf] = 0.7
    H = DiscreteNode(:B, cpt_h)

    @test_throws ErrorException("network nodes names must be unique") CredalNetwork([F, B, L, D, H])


    cpt_h = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:D, :H])
    cpt_h[:D=>:Dt, :H=>:Bt] = 0.6
    cpt_h[:D=>:Dt, :H=>:Bf] = 0.4
    cpt_h[:D=>:Df, :H=>:Bt] = 0.3
    cpt_h[:D=>:Df, :H=>:Bf] = 0.7
    H = DiscreteNode(:H, cpt_h)
    @test_throws ErrorException("network nodes states must be unique") CredalNetwork([F, B, L, D, H])

    cpt_h = DiscreteConditionalProbabilityTable{PreciseDiscreteProbability}([:D, :H])
    cpt_h[:D=>:Dt, :H=>:Ht] = 0.6
    cpt_h[:D=>:Dt, :H=>:Hf] = 0.4
    cpt_h[:D=>:Df, :H=>:Ht] = 0.3
    cpt_h[:D=>:Df, :H=>:Hf] = 0.7
    H = DiscreteNode(:H, cpt_h)
    @test_throws ErrorException("all nodes are precise. Use BayesianNetwork structure!") CredalNetwork([F, B, L, D, H])


    H = DiscreteNode(:H, cpt_h, Dict(:Ht => [Parameter(1, :H)], :Hf => [Parameter(0, :H)]))

    cpt_i = ContinuousConditionalProbabilityTable{PreciseContinuousInput}()
    cpt_i[] = Normal()
    I = ContinuousNode(:I, cpt_i)

    @test_throws ErrorException("node/s [:I] are continuous. Use EnhancedBayesianNetwork structure!") CredalNetwork([F, B, L, D, H, I])


    cpt_h = DiscreteConditionalProbabilityTable{ImpreciseDiscreteProbability}([:D, :H])
    cpt_h[:D=>:Dt, :H=>:Ht] = (0.6, 0.8)
    cpt_h[:D=>:Dt, :H=>:Hf] = (0.2, 0.4)
    cpt_h[:D=>:Df, :H=>:Ht] = (0.2, 0.3)
    cpt_h[:D=>:Df, :H=>:Hf] = (0.7, 0.8)
    H = DiscreteNode(:H, cpt_h)

    cn = CredalNetwork([F, B, L, D, H])

    @test cn.A == sparse(zeros(5, 5))
    @test cn.topology == Dict(:F => 1, :H => 5, :D => 4, :B => 2, :L => 3)
    @test issetequal(cn.nodes, [F, B, L, D, H])

    add_child!(cn, F, L)
    add_child!(cn, F, D)
    add_child!(cn, B, D)
    add_child!(cn, D, H)
    order!(cn)

    @test cn.A == sparse([
        0.0 0.0 1.0 1.0 0.0;
        0.0 0.0 0.0 1.0 0.0;
        0.0 0.0 0.0 0.0 0.0;
        0.0 0.0 0.0 0.0 1.0;
        0.0 0.0 0.0 0.0 0.0])
    @test cn.topology == Dict(:F => 1, :H => 5, :D => 4, :B => 2, :L => 3)
    @test issetequal(cn.nodes, [F, B, L, D, H])

    ebn = EnhancedBayesianNetwork([F, B, L, D, H])
    add_child!(ebn, F, L)
    add_child!(ebn, F, D)
    add_child!(ebn, B, D)
    add_child!(ebn, D, H)
    order!(ebn)

    @test cn == CredalNetwork(ebn)
end