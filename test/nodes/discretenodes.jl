@testset "Discrete Nodes" begin
    @testset "structure and setindex" begin
        node_a = DiscreteNode(:a)
        @test isa(node_a, AbstractNode)
        @test isa(node_a, AbstractDiscreteNode)
        @test isa(node_a, DiscreteNode)
        @test isa(node_a.cpt, ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability})
        @test names(node_a.cpt.data) == ["a", "Π"]
        @test isa(node_a.parameters, Vector{Pair{Symbol,Vector{Parameter}}})
        @test isa(node_a.results, Dict{Vector{Symbol},Tuple})
        node_c = DiscreteNode(:c, [:a])
        @test isa(node_c, DiscreteNode)
        @test isa(node_c.cpt, ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability})
        @test names(node_c.cpt.data) == ["a", "c", "Π"]
        @test isa(node_c.parameters, Vector{Pair{Symbol,Vector{Parameter}}})
        @test isa(node_c.results, Dict{Vector{Symbol},Tuple})

        node_a[:a=>:a1] = Interval(0.1, 0.3)
        node_a[:a=>:a2] = Interval(0.6, 0.8)
        node_a[:a=>:a3] = 0.2
        @test node_a.cpt.data.a == [:a1, :a2, :a3]
        @test node_a.cpt.data.Π[1] == Interval(0.1, 0.3)
        @test node_a.cpt.data.Π[2] == Interval(0.6, 0.8)
        @test node_a.cpt.data.Π[3] == 0.2
        node_c[:a=>:a1, :c=>:c1] = 0.3
        node_c[:a=>:a1, :c=>:c2] = 0.7
        node_c[:a=>:a2, :c=>:c1] = Interval(0.3, 0.6)
        node_c[:a=>:a2, :c=>:c2] = Interval(0.3, 0.6)
        @test isa(node_c, AbstractNode)
        @test isa(node_c, AbstractDiscreteNode)
        @test isa(node_c, DiscreteNode)
        @test node_c.cpt.data.a == [:a1, :a1, :a2, :a2]
        @test node_c.cpt.data.c == [:c1, :c2, :c1, :c2]
        @test node_c.cpt.data.Π[1] == 0.3
        @test node_c.cpt.data.Π[2] == 0.7
        @test node_c.cpt.data.Π[3] == Interval(0.3, 0.6)
        @test node_c.cpt.data.Π[4] == Interval(0.3, 0.6)

        ## tests for parameters    
        parameters = [:a1 => [Parameter(1, :A)], :a2 => [Parameter(0, :A)]]
        results = Dict{Vector{Symbol},Tuple}()
        node_a = DiscreteNode(:a, parameters)
        node_a[:a=>:a1] = 0.2
        @test node_a.cpt.data.a == [:a1]
        @test node_a.cpt.data.Π == [0.2]
        @test node_a.parameters == parameters
        node_c = DiscreteNode(:c, [:a], parameters)
        node_c[:a=>:a1, :c=>:c1] = 0.2
        @test node_c.cpt.data.c == [:c1]
        @test node_c.cpt.data.a == [:a1]
        @test node_c.cpt.data.Π == [0.2]
        @test node_c.parameters == parameters

        ## tests for nodes with a pre-defined CPT
        cpt_a = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}(:a)
        cpt_a[:a=>:a1] = 0.2
        cpt_a[:a=>:a2] = 0.8
        parameters = [:a1 => [Parameter(1, :A)], :a2 => [Parameter(0, :A)]]
        results = Dict{Vector{Symbol},Tuple}()
        node_a = DiscreteNode(cpt_a)
        @test node_a.cpt == cpt_a
        node_a = DiscreteNode(cpt_a, parameters)
        @test node_a.cpt == cpt_a
        @test node_a.parameters == parameters
        node_a = DiscreteNode(cpt_a, parameters, results)
        @test node_a.cpt == cpt_a
        @test node_a.parameters == parameters
        @test node_a.results == results

        cpt_c = ConditionalProbabilityTable{EnhancedBayesianNetworks.DiscreteProbability}([:a, :c])
        cpt_c[:a=>:a1, :c=>:c1] = 0.2
        cpt_c[:a=>:a1, :c=>:c2] = 0.8
        parameters = [:c1 => [Parameter(1, :C)], :c2 => [Parameter(0, :C)]]
        results = Dict{Vector{Symbol},Tuple}()
        node_c = DiscreteNode(cpt_c)
        @test node_c.cpt == cpt_c
        node_c = DiscreteNode(cpt_c, parameters)
        @test node_c.cpt == cpt_c
        @test node_c.parameters == parameters
        node_c = DiscreteNode(cpt_c, parameters, results)
        @test node_c.cpt == cpt_c
        @test node_c.parameters == parameters
        @test node_c.results == results
    end

    @testset "main functions" begin
        @test_throws ErrorException(":Π is not allowed as node name") DiscreteNode(:Π)
        node_a = DiscreteNode(:a)
        node_a[:a=>:a1] = Interval(0.1, 0.3)
        node_a[:a=>:a2] = Interval(0.6, 0.8)
        node_a[:a=>:a3] = 0.2
        node_c = DiscreteNode(:c, [:a])
        node_c[:a=>:a1, :c=>:c1] = 0.3
        node_c[:a=>:a1, :c=>:c2] = 0.7
        node_c[:a=>:a2, :c=>:c1] = Interval(0.3, 0.6)
        node_c[:a=>:a2, :c=>:c2] = Interval(0.3, 0.6)
        node_b = DiscreteNode(:b, [:a])
        node_b[:a=>:a1, :b=>:b1] = 0.3
        node_b[:a=>:a1, :b=>:b2] = 0.7
        node_b[:a=>:a2, :b=>:b1] = 0.6
        node_b[:a=>:a2, :b=>:b2] = 0.4

        @test issetequal(states(node_a), [:a1, :a2, :a3])
        @test issetequal(states(node_c), [:c1, :c2])
        sc_a = [[:a => :a1], [:a => :a2], [:a => :a3]]
        sc_c = [[:a => :a1, :c => :c1], [:a => :a1, :c => :c2], [:a => :a2, :c => :c1], [:a => :a2, :c => :c2]]
        @test scenarios(node_a) == sc_a
        @test scenarios(node_c) == sc_c
        @test isprecise(node_a) == false
        @test isprecise(node_c) == false
        @test isprecise(node_b)
        @test isroot(node_a)
        @test isroot(node_c) == false
        @test isempty(parents(node_a))
        @test issetequal(parents(node_c), [:a])

        ## inputs function
        parameters = [:a1 => [Parameter(1, :A)], :a2 => [Parameter(0, :A)], :a3 => [Parameter(-1, :A)]]
        node_a = DiscreteNode(:a, parameters)
        node_a[:a=>:a1] = Interval(0.1, 0.3)
        node_a[:a=>:a2] = Interval(0.6, 0.8)
        node_a[:a=>:a3] = 0.2

        evidence = Evidence(:b => :a1)
        @test_throws ErrorException("evidence `Dict(:b => :a1)` does not contain the node `a`") EnhancedBayesianNetworks._inputs(node_a, evidence)
        evidence = Evidence(:a => :a4)
        @test_throws ErrorException("evidence `Dict(:a => :a4)` contains a not existing state `a4` for node `a`") EnhancedBayesianNetworks._inputs(node_a, evidence)
        evidence = Evidence(:a => :a1)
        @test EnhancedBayesianNetworks._inputs(node_a, evidence) == [Parameter(1, :A)]
        evidence = Evidence(:a => :a2, :b => :b1)
        @test EnhancedBayesianNetworks._inputs(node_a, evidence) == [Parameter(0, :A)]
    end

    @testset "extreme points" begin
        int1 = Interval(0.2, 0.5)
        int2 = Interval(0.5, 0.6)
        int3 = Interval(0.2, 0.4)
        extreme_probs = EnhancedBayesianNetworks._extreme_probabilities(int1, int2, int3)
        @test all(isapprox.(extreme_probs[1], [0.2, 0.6, 0.2]))
        @test all(isapprox.(extreme_probs[2], [0.3, 0.5, 0.2]))
        @test all(isapprox.(extreme_probs[3], [0.2, 0.5, 0.3]))

        node_a = DiscreteNode(:a)
        node_a[:a=>:a1] = Interval(0.4, 0.5)
        node_a[:a=>:a2] = Interval(0.3, 0.4)
        node_a[:a=>:a3] = Interval(0.1, 0.2)
        nodes = EnhancedBayesianNetworks._extreme_nodes(node_a)
        @test all([i.name == node_a.name for i in nodes])
        @test all([i.parameters == node_a.parameters for i in nodes])
        @test all([i.results == node_a.results for i in nodes])
        @test nodes[1].cpt.data.Π == [0.4, 0.4, 0.2]
        @test nodes[2].cpt.data.Π == [0.5, 0.3, 0.2]
        @test nodes[3].cpt.data.Π == [0.5, 0.4, 0.1]

        node_a = DiscreteNode(:a, [:b, :c])
        node_a[:b=>:b1, :c=>:c1, :a=>:a1] = Interval(0.1, 0.2)
        node_a[:b=>:b1, :c=>:c1, :a=>:a2] = Interval(0.3, 0.7)
        node_a[:b=>:b1, :c=>:c1, :a=>:a3] = Interval(0.4, 0.5)
        node_a[:b=>:b1, :c=>:c2, :a=>:a1] = Interval(0.15, 0.45)
        node_a[:b=>:b1, :c=>:c2, :a=>:a2] = Interval(0.05, 0.25)
        node_a[:b=>:b1, :c=>:c2, :a=>:a3] = Interval(0.45, 0.55)
        node_a[:b=>:b2, :c=>:c1, :a=>:a1] = Interval(0.01, 0.02)
        node_a[:b=>:b2, :c=>:c1, :a=>:a2] = Interval(0.03, 0.07)
        node_a[:b=>:b2, :c=>:c1, :a=>:a3] = Interval(0.93, 0.99)
        node_a[:b=>:b2, :c=>:c2, :a=>:a1] = Interval(0.1112, 0.21123)
        node_a[:b=>:b2, :c=>:c2, :a=>:a2] = Interval(0.31123, 0.71123)
        node_a[:b=>:b2, :c=>:c2, :a=>:a3] = Interval(0.41123, 0.511223)
        nodes = EnhancedBayesianNetworks._extreme_nodes(node_a)
        @test length(nodes) == 400
    end
end