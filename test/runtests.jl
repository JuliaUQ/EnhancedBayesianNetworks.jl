using Test
using TestItems
using TestItemRunner
using EnhancedBayesianNetworks

# Add dependencies only needed for testing
@testsnippet ExtraDeps begin
    using CSV
    using DataFrames
    using SparseArrays
    using Suppressor
end

# Helper function to check that the indices of the nodes in the network are coherent with the topology and adjacency matrix
@testsnippet CheckSetup begin
    function check_index_coherence(net)
        @test length(net.topology) == length(net.nodes) == size(net.A, 1)
        @test all(net.topology[node.name] == i for (i, node) in enumerate(net.nodes))
    end
end

@run_package_tests()
