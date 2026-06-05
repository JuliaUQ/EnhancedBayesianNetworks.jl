@testitem "Inference State - BN" begin
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

    nodes = [v, s, t, l]
    bn = BayesianNetwork(nodes)
    query = [:L]

    evidence = Evidence(:V => :yesV, :T => :yesT, :G => :g1)
    @test_throws ErrorException("Invalid Evidence: evidence [:T => :yesT, :G => :g1, :V => :yesV] contains Symbols [:G] that are not associated to any node of the network") InferenceState(bn, query, evidence)
    @test_throws ErrorException("Invalid Evidence: evidence [:T => :yesT, :G => :g1, :V => :yesV] contains Symbols [:G] that are not associated to any node of the network") EnhancedBayesianNetworks.verify_evidence(evidence, bn)

    evidence = Evidence(:V => :yesV, :T => :T1)
    @test_throws ErrorException("Invalid Evidence: evidence [:T => :T1, :V => :yesV] defines state :T1 for node :T that does not belong to its possible states [:yesT, :noT]") InferenceState(bn, query, evidence)
    @test_throws ErrorException("Invalid Evidence: evidence [:T => :T1, :V => :yesV] defines state :T1 for node :T that does not belong to its possible states [:yesT, :noT]") EnhancedBayesianNetworks.verify_evidence(evidence, bn)

    evidence = Evidence(:V => :yesV, :T => :yesT)
    query = [:L, :R]
    @test_throws ErrorException("Invalid Query: queried nodes vector [:L, :R] contains Symbols [:R] that are not associated to any node of the network") InferenceState(bn, query, evidence)
    @test_throws ErrorException("Invalid Query: queried nodes vector [:L, :R] contains Symbols [:R] that are not associated to any node of the network") EnhancedBayesianNetworks.verify_query(query, bn, evidence)

    query = [:L, :T]
    @test_throws ErrorException("Invalid Query: queried nodes vector [:L, :T] contains Symbols [:T] that are already part of the evidence [:T => :yesT, :V => :yesV]") InferenceState(bn, query, evidence)
    @test_throws ErrorException("Invalid Query: queried nodes vector [:L, :T] contains Symbols [:T] that are already part of the evidence [:T => :yesT, :V => :yesV]") EnhancedBayesianNetworks.verify_query(query, bn, evidence)
end

@testitem "Credal Networks" begin
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

    g = DiscreteNode(:G)
    g[:G=>:g1] = Interval(0.1, 0.2)
    g[:G=>:g2] = Interval(0.15, 0.25)
    g[:G=>:g3] = 0.2
    g[:G=>:g4] = Interval(0.3, 0.5)

    nodes = [v, s, t, l, g]
    cn = CredalNetwork(nodes)
    query = [:L]

    evidence = Evidence(:V => :yesV, :T => :yesT, :M => :g1)
    @test_throws ErrorException("Invalid Evidence: evidence [:T => :yesT, :M => :g1, :V => :yesV] contains Symbols [:M] that are not associated to any node of the network") InferenceState(cn, query, evidence)

    evidence = Evidence(:V => :yesV, :T => :T1)
    @test_throws ErrorException("Invalid Evidence: evidence [:T => :T1, :V => :yesV] defines state :T1 for node :T that does not belong to its possible states [:yesT, :noT]") InferenceState(cn, query, evidence)

    evidence = Evidence(:V => :yesV, :T => :yesT)
    query = [:L, :R]
    @test_throws ErrorException("Invalid Query: queried nodes vector [:L, :R] contains Symbols [:R] that are not associated to any node of the network") InferenceState(cn, query, evidence)

    query = [:L, :T]
    @test_throws ErrorException("Invalid Query: queried nodes vector [:L, :T] contains Symbols [:T] that are already part of the evidence [:T => :yesT, :V => :yesV]") InferenceState(cn, query, evidence)
end