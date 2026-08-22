@testitem "Show - _topology_parents" setup = [SetupShowNet] begin
    @test _topology_parents(ebn, D) == "-"
    @test _topology_parents(ebn, cf) == "D"
end

@testitem "Show - EnhancedBayesianNetwork" setup = [SetupShowNet] begin
    @test sprint(show, ebn) == "EnhancedBayesianNetwork(2 nodes)"
    s = plainshow(ebn)
    @test occursin("EnhancedBayesianNetwork", s)
    @test occursin("Nodes: 2", s)
    @test occursin("Edges: 1", s)
    @test occursin("Continuous nodes: 2", s)
    @test occursin("Functional nodes: 1", s)
    @test occursin("Topology:", s)
    @test occursin("ContinuousFunctional", s)
end

@testitem "Show - BayesianNetwork" setup = [SetupShowNet] begin
    @test sprint(show, bn) == "BayesianNetwork(3 nodes)"
    s = plainshow(bn)
    @test occursin("BayesianNetwork", s)
    @test occursin("Nodes: 3", s)
    @test occursin("Edges: 2", s)
    @test occursin("sunny, cloudy", s)
    @test occursin("Topology:", s)
end

@testitem "Show - CredalNetwork" setup = [SetupShowNet] begin
    @test sprint(show, cn) == "CredalNetwork(3 nodes)"
    s = plainshow(cn)
    @test occursin("CredalNetwork", s)
    @test occursin("Nodes: 3", s)
    @test occursin("Precise nodes: 1", s)
    @test occursin("Credal nodes: 2", s)
    @test occursin("Tampering, Fire", s)
end
