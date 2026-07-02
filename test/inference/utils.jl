@testsnippet SetupModifiedSprinklerBN begin
    weather = DiscreteNode(:W)
    weather[:W=>:Cloudy] = 0.5
    weather[:W=>:Sunny] = 0.5
    rain = DiscreteNode(:R, [:W])
    rain[:W=>:Cloudy, :R=>:Yes] = 0.8
    rain[:W=>:Cloudy, :R=>:No] = 0.2
    rain[:W=>:Sunny, :R=>:Yes] = 0.1
    rain[:W=>:Sunny, :R=>:No] = 0.9
    sprinkler = DiscreteNode(:S, [:W])
    sprinkler[:W=>:Cloudy, :S=>:On] = 0.4
    sprinkler[:W=>:Cloudy, :S=>:Off] = 0.4
    sprinkler[:W=>:Cloudy, :S=>:broken] = 0.2
    sprinkler[:W=>:Sunny, :S=>:On] = 0.6
    sprinkler[:W=>:Sunny, :S=>:Off] = 0.3
    sprinkler[:W=>:Sunny, :S=>:broken] = 0.1
    grass = DiscreteNode(:G, [:S, :R])
    grass[:R=>:No, :S=>:On, :G=>:Dry] = 0.2
    grass[:R=>:No, :S=>:On, :G=>:Wet] = 0.8
    grass[:R=>:No, :S=>:Off, :G=>:Wet] = 0.2
    grass[:R=>:No, :S=>:Off, :G=>:Dry] = 0.8
    grass[:R=>:No, :S=>:broken, :G=>:Wet] = 0.1
    grass[:R=>:No, :S=>:broken, :G=>:Dry] = 0.9
    grass[:R=>:Yes, :S=>:On, :G=>:Wet] = 0.6
    grass[:R=>:Yes, :S=>:On, :G=>:Dry] = 0.4
    grass[:R=>:Yes, :S=>:Off, :G=>:Wet] = 0.55
    grass[:R=>:Yes, :S=>:Off, :G=>:Dry] = 0.45
    grass[:R=>:Yes, :S=>:broken, :G=>:Wet] = 0.58
    grass[:R=>:Yes, :S=>:broken, :G=>:Dry] = 0.42

    nodes = [weather, rain, sprinkler, grass]
    bn = BayesianNetwork(nodes)
    add_child!(bn, :W, :R)
    add_child!(bn, :W, :S)
    add_child!(bn, :R, :G)
    add_child!(bn, :S, :G)
end

@testitem "Inference - verify query & evidence" setup=[ExtraDeps, SetupModifiedSprinklerBN] begin
    @suppress order!(bn)

    @test_throws ErrorException("Invalid Query: queried nodes vector [:H] contains Symbols [:H] that are not associated to any node of the network") EnhancedBayesianNetworks.verify_query([:H], bn, Evidence(:G=>:Wet))
    @test_throws ErrorException("Invalid Query: queried nodes vector [:G] contains Symbols [:G] that are already part of the evidence [:G => :Wet]") EnhancedBayesianNetworks.verify_query([:G], bn, Evidence(:G=>:Wet))

    @test_throws ErrorException("Invalid Evidence: evidence [:H => :Wet] contains Symbols [:H] that are not associated to any node of the network") EnhancedBayesianNetworks.verify_evidence(Evidence(:H=>:Wet), bn)
    @test_throws ErrorException("Invalid Evidence: evidence [:G => :Dirty] defines state :Dirty for node :G that does not belong to its possible states [:Dry, :Wet]") EnhancedBayesianNetworks.verify_evidence(Evidence(:G=>:Dirty), bn)
end

@testitem "Inference - query & evidence to index" begin
    idx_to_node = [:W, :R, :S, :G]
    idx_to_state = [[:Cloudy, :Sunny], [:Yes, :No], [:On, :Off, :broken], [:Dry, :Wet]]
    node_to_idx = Dict(:R => 2, :G => 4, :W => 1, :S => 3)
    state_to_idx = [Dict(:Cloudy => 1, :Sunny => 2), Dict(:No => 2, :Yes => 1), Dict(:broken => 3, :Off => 2, :On => 1), Dict(:Dry => 1, :Wet => 2)]
    ns = EnhancedBayesianNetworks.NetworkSchema(node_to_idx, idx_to_node, state_to_idx, idx_to_state)

    @test EnhancedBayesianNetworks.query_to_idx(:W, ns) == [1]
    @test EnhancedBayesianNetworks.query_to_idx([:W, :S], ns) == [1, 3]

    @test EnhancedBayesianNetworks.evidence_to_idx(Evidence(:W=>:Sunny), ns) == [(1, 2)]
    @test issetequal(EnhancedBayesianNetworks.evidence_to_idx(Evidence(:W=>:Sunny, :G=>:Wet), ns), [(1, 2), (4, 2)])
end