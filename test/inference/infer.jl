@testsnippet SetupAsiaBN begin
    V = DiscreteNode(:V)
    V[:V=>:YesV] = 0.01
    V[:V=>:NoV] = 0.99
    S = DiscreteNode(:S)
    S[:S=>:YesS] = 0.01
    S[:S=>:NoS] = 0.99
    T = DiscreteNode(:T, [:V])
    T[:V=>:YesV, :T=>:YesT] = 0.05
    T[:V=>:YesV, :T=>:NoT] = 0.95
    T[:V=>:NoV, :T=>:YesT] = 0.01
    T[:V=>:NoV, :T=>:NoT] = 0.99
    L = DiscreteNode(:L, [:S])
    L[:S=>:YesS, :L=>:YesL] = 0.1
    L[:S=>:YesS, :L=>:NoL] = 0.9
    L[:S=>:NoS, :L=>:YesL] = 0.01
    L[:S=>:NoS, :L=>:NoL] = 0.99
    B = DiscreteNode(:B, [:S])
    B[:S=>:YesS, :B=>:YesB] = 0.6
    B[:S=>:YesS, :B=>:NoB] = 0.4
    B[:S=>:NoS, :B=>:YesB] = 0.3
    B[:S=>:NoS, :B=>:NoB] = 0.7
    E = DiscreteNode(:E, [:L, :T])
    E[:L=>:YesL, :T=>:YesT, :E=>:YesE] = 1
    E[:L=>:YesL, :T=>:YesT, :E=>:NoE] = 0
    E[:L=>:YesL, :T=>:NoT, :E=>:YesE] = 1
    E[:L=>:YesL, :T=>:NoT, :E=>:NoE] = 0
    E[:L=>:NoL, :T=>:YesT, :E=>:YesE] = 1
    E[:L=>:NoL, :T=>:YesT, :E=>:NoE] = 0
    E[:L=>:NoL, :T=>:NoT, :E=>:YesE] = 0
    E[:L=>:NoL, :T=>:NoT, :E=>:NoE] = 1
    D = DiscreteNode(:D, [:B, :E])
    D[:B=>:YesB, :E=>:YesE, :D=>:YesD] = 0.9
    D[:B=>:YesB, :E=>:YesE, :D=>:NoD] = 0.1
    D[:B=>:YesB, :E=>:NoE, :D=>:YesD] = 0.8
    D[:B=>:YesB, :E=>:NoE, :D=>:NoD] = 0.2
    D[:B=>:NoB, :E=>:YesE, :D=>:YesD] = 0.7
    D[:B=>:NoB, :E=>:YesE, :D=>:NoD] = 0.3
    D[:B=>:NoB, :E=>:NoE, :D=>:YesD] = 0.1
    D[:B=>:NoB, :E=>:NoE, :D=>:NoD] = 0.9
    X = DiscreteNode(:X, [:E])
    X[:E=>:YesE, :X=>:YesX] = 0.98
    X[:E=>:YesE, :X=>:NoX] = 0.02
    X[:E=>:NoE, :X=>:YesX] = 0.05
    X[:E=>:NoE, :X=>:NoX] = 0.95
    nodes = [V, S, T, L, B, E, D, X]
    bn = BayesianNetwork(nodes)
    add_child!(bn, V, T)
    add_child!(bn, S, [L, B])
    add_child!(bn, [T, L], E)
    add_child!(bn, [E, B], D)
    add_child!(bn, E, X)
    order!(bn)
end

@testsnippet SetupFireProtectionCN begin
    T = DiscreteNode(:Tampering)
    T[:Tampering=>:YesT] = 0.98
    T[:Tampering=>:NoT] = 0.02
    F = DiscreteNode(:Fire)
    F[:Fire=>:YesF] = Interval(0.98, 0.99)
    F[:Fire=>:NoF] = Interval(0.01, 0.02)
    A = DiscreteNode(:Alarm, [:Tampering, :Fire])
    A[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.4, 0.6)
    A[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.4, 0.5)
    A[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.85, 0.9)
    A[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.1, 0.15)
    A[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.985, 0.99)
    A[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.01, 0.015)
    A[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.0001, 0.0002)
    A[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.9998, 0.9999)
    S = DiscreteNode(:Smoke, [:Fire])
    S[:Fire=>:YesF, :Smoke=>:YesS] = Interval(0.87, 0.91)
    S[:Fire=>:YesF, :Smoke=>:NoS] = Interval(0.09, 0.13)
    S[:Fire=>:NoF, :Smoke=>:YesS] = Interval(0.01, 0.1)
    S[:Fire=>:NoF, :Smoke=>:NoS] = Interval(0.9, 0.99)
    L = DiscreteNode(:Leaving, [:Alarm])
    L[:Alarm=>:YesA, :Leaving=>:YesL] = Interval(0.88, 0.99)
    L[:Alarm=>:YesA, :Leaving=>:NoL] = Interval(0.001, 0.42)
    L[:Alarm=>:NoA, :Leaving=>:YesL] = Interval(0.1, 0.12)
    L[:Alarm=>:NoA, :Leaving=>:NoL] = Interval(0.58, 0.99)
    R = DiscreteNode(:Report, [:Leaving])
    R[:Leaving=>:YesL, :Report=>:YesR] = Interval(0.25, 0.76)
    R[:Leaving=>:YesL, :Report=>:NoR] = Interval(0.24, 0.75)
    R[:Leaving=>:NoL, :Report=>:YesR] = Interval(0.01, 0.2)
    R[:Leaving=>:NoL, :Report=>:NoR] = Interval(0.8, 0.99)
    nodes = [T, F, A, S, L, R]
    cn = CredalNetwork(nodes)
    add_child!(cn, :Tampering, :Alarm)
    add_child!(cn, :Fire, :Alarm)
    add_child!(cn, :Fire, :Smoke)
    add_child!(cn, :Alarm, :Leaving)
    add_child!(cn, :Leaving, :Report)
    order!(cn)
end

@testitem "Inference - Posterior" setup=[SetupAsiaBN] begin
    posterior = infer(bn, [:X], Evidence())
    @test posterior isa Posterior

    posterior = infer(bn, Symbol[], Evidence())
    @test posterior isa Posterior

    posterior = infer(bn, Symbol[], Evidence(:D=>:YesD))
    @test posterior isa Posterior

    posterior = infer(bn, Symbol[:X, :E], Evidence(:D=>:YesD))
    @test posterior isa Posterior
end

@testitem "Inference - CredalPosterior" begin
    idx_to_node = [:W]
    idx_to_state = [[:Cloudy, :Sunny]]
    node_to_idx = Dict(:W => 1)
    state_to_idx = [Dict(:Cloudy => 1, :Sunny => 2)]

    ns = EnhancedBayesianNetworks.NetworkSchema(
        node_to_idx,
        idx_to_node,
        state_to_idx,
        idx_to_state
    )

    f1 = EnhancedBayesianNetworks.Factor([1], [0.4, 0.6])
    f2 = EnhancedBayesianNetworks.Factor([1], [0.3, 0.7])
    p = EnhancedBayesianNetworks.Posterior(f1, ns, [:W], Evidence())
    cp = EnhancedBayesianNetworks.CredalPosterior([p], f2, f1, ns, [:W], Evidence())

    @test length(cp.posteriors) == 1
    @test cp.lower === f2
    @test cp.upper === f1
    @test cp.schema === ns
    @test cp.query == [:W]
    @test isempty(cp.evidence)
end

@testitem "Inference - extreme bns" setup=[SetupFireProtectionCN] begin
    nodes = [T, F, A, S, L, R]
    cn = CredalNetwork(nodes)
    bns = EnhancedBayesianNetworks._extreme_bayesian_networks(cn)
    @test length(bns) == 2048
    for bn in bns
        @test length(bn.nodes) == length(cn.nodes)
        @test bn.A == cn.A
        @test bn.topology == cn.topology
    end
end

@testitem "Inference - Infer" setup=[ExtraDeps, SetupAsiaBN, SetupFireProtectionCN] begin

    p = infer(bn, [:V], Evidence())
    @test p.factor.table ≈ [0.01, 0.99]

    p = infer(bn, [:S], Evidence())
    @test p.factor.table ≈ [0.01, 0.99]

    p = infer(bn, [:T], Evidence())
    @test p.factor.table[1] ≈ 0.0104
    @test p.factor.table[2] ≈ 0.9896

    p = infer(bn, [:E], Evidence())
    @test p.factor.table[1] ≈ 0.02118664 atol=1e-8
    @test p.factor.table[2] ≈ 0.97881336 atol=1e-8

    p = infer(bn, [:E], Evidence())
    @test p.factor.table[1] ≈ 0.02118664 atol=1e-8
    @test p.factor.table[2] ≈ 0.97881336 atol=1e-8

    p = infer(bn, [:T], Evidence(:E => :NoE))
    @test p.factor.table[1] ≈ 0.0
    @test p.factor.table[2] ≈ 1.0


    @test_throws ErrorException("Invalid Query: queried nodes vector [:E] contains Symbols [:E] that are already part of the evidence [:E => :YesE]") infer(bn, [:E], Evidence(:E => :YesE))

    p = infer(bn, [:T, :L], Evidence())
    @test size(p.factor.table) == (2, 2)
    @test sum(p.factor.table) ≈ 1.0

    p1 = infer(bn, [:D], Evidence(:X => :YesX), fill_score)
    p2 = infer(bn, [:D], Evidence(:X => :YesX), factor_score)
    p3 = infer(bn, [:D], Evidence(:X => :YesX), fill_factor_score)

    @test p1.factor.table ≈ p2.factor.table
    @test p1.factor.table ≈ p3.factor.table

    @test_throws ErrorException("Invalid Query: queried nodes vector [:T, :E] contains Symbols [:E] that are already part of the evidence [:E => :YesE]") infer(bn, [:T, :E], Evidence(:E => :YesE))

    # Credal Inference
    p = @suppress infer(cn, [:Fire], Evidence())
    @test p.lower.table[1] ≈ 0.98
    @test p.upper.table[1] ≈ 0.99
    @test p.lower.table[2] ≈ 0.01
    @test p.upper.table[2] ≈ 0.02

    @test_throws ErrorException("Invalid Query: queried nodes vector [:Alarm] contains Symbols [:Alarm] that are already part of the evidence [:Alarm => :YesA]") infer(cn, [:Alarm], Evidence(:Alarm => :YesA))

    @test_throws ErrorException("Invalid Query: queried nodes vector [:Fire, :Alarm] contains Symbols [:Alarm] that are already part of the evidence [:Alarm => :YesA]") infer(cn, [:Fire, :Alarm], Evidence(:Alarm => :YesA))

    p = @suppress infer(cn, [:Fire], Evidence(:Smoke => :YesS))
    @test p.lower.table[1] ≈ 0.997659723847414
    @test p.upper.table[1] ≈ 0.9998890122086571

    p = @suppress infer(cn, [:Fire], Evidence(:Smoke => :YesS, :Alarm => :YesA))
    @test p.lower.table[1] ≈ 0.99595720920615
    @test p.upper.table[1] ≈ 0.9998478952774653

end