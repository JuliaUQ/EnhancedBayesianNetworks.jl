@testitem "Inference - NetworkSchema" setup=[ExtraDeps, SetupBNgrass] begin

    @suppress order!(bn)

    idx_to_node = [:W, :R, :S, :G]
    idx_to_state = [[:Cloudy, :Sunny], [:Yes, :No], [:On, :Off, :broken], [:Dry, :Wet]]
    node_to_idx = Dict(:R => 2, :G => 4, :W => 1, :S => 3)
    state_to_idx = [Dict(:Cloudy => 1, :Sunny => 2), Dict(:No => 2, :Yes => 1), Dict(:broken => 3, :Off => 2, :On => 1), Dict(:Dry => 1, :Wet => 2)]

    ns1 = EnhancedBayesianNetworks.NetworkSchema(node_to_idx, idx_to_node, state_to_idx, idx_to_state)
    ns2 = EnhancedBayesianNetworks.NetworkSchema(bn)

    @test ns1.state_to_idx == state_to_idx
    @test ns1.idx_to_state == idx_to_state
    @test ns1.node_to_idx == node_to_idx
    @test ns1.idx_to_node == idx_to_node

    @test ns2.state_to_idx == state_to_idx
    @test ns2.idx_to_state == idx_to_state
    @test ns2.node_to_idx == node_to_idx
    @test ns2.idx_to_node == idx_to_node
end

@testitem "Inference - InteractionGraph" setup=[ExtraDeps, SetupBN] begin

    ig = EnhancedBayesianNetworks.InteractionGraph(bn)
    @test ig.neighbors[1] == Set([3])
    @test ig.neighbors[2] == Set([5, 4])
    @test ig.neighbors[3] == Set([4, 6, 1])
    @test ig.neighbors[4] == Set([6, 2, 3])
    @test ig.neighbors[5] == Set([6, 7, 2])
    @test ig.neighbors[6] == Set([5, 4, 7, 8, 3])
    @test ig.neighbors[7] == Set([5, 6])
    @test ig.neighbors[8] == Set([6])
end