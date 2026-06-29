@testitem "Dispatch Function" setup=[ExtraDeps] begin

    r = ContinuousNode(:R, Normal())

    v = DiscreteNode(:V)
    v[:V=>:yesV] = 0.01
    v[:V=>:noV] = 0.99

    s = DiscreteNode(:S)
    s[:S=>:yesS] = 0.5
    s[:S=>:noS] = 0.5

    t = DiscreteNode(:T, [:V])
    t[:V=>:yesV, :T=>:yesT] = 0.05
    t[:V=>:yesV, :T=>:noT] = 0.95
    t[:V=>:noV, :T=>:yesT] = 0.01
    t[:V=>:noV, :T=>:noT] = 0.99

    l = DiscreteNode(:L, [:S])
    l[:S=>:yesS, :L=>:yesL] = 0.1
    l[:S=>:yesS, :L=>:noL] = 0.9
    l[:S=>:noS, :L=>:yesL] = 0.01
    l[:S=>:noS, :L=>:noL] = 0.99

    f1 = DiscreteFunctionalNode(
        :F1, [Model(df -> df.v1 .+ df.R, :f1)], df -> 0.8 .- df.f1, MonteCarlo(200)
    )

    g = DiscreteNode(:G)
    g[:G=>:g1] = Interval(0.1, 0.2)
    g[:G=>:g2] = Interval(0.15, 0.25)
    g[:G=>:g3] = 0.2
    g[:G=>:g4] = Interval(0.3, 0.5)

    h = DiscreteNode(:H)
    h[:H=>:yesH] = 0.01
    h[:H=>:noH] = 0.99

    nodes = [r, v, s, t, g, h]
    ebn = EnhancedBayesianNetwork(nodes)
    @test isa(EnhancedBayesianNetworks.dispatch(ebn), EnhancedBayesianNetwork)

    nodes = [v, s, t, g, h]
    ebn = EnhancedBayesianNetwork(nodes)
    @test isa(EnhancedBayesianNetworks.dispatch(ebn), CredalNetwork)

    nodes = [v, s, t]
    ebn = EnhancedBayesianNetwork(nodes)
    @test isa(EnhancedBayesianNetworks.dispatch(ebn), BayesianNetwork)

    nodes = [v, s, t, g]
    cn = CredalNetwork(nodes)
    @test isa(EnhancedBayesianNetworks.dispatch(cn), CredalNetwork)

    nodes = [v, s, t]
    cn = @suppress CredalNetwork(nodes)
    @test isa(EnhancedBayesianNetworks.dispatch(cn), BayesianNetwork)
end