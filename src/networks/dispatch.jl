# Pick the concrete network type for a set of already-discrete nodes: a BayesianNetwork when every node is precise, a CredalNetwork when some are imprecise.
function _dispatch(nodes::AbstractVector{DiscreteNode}, topology::Dict, A::SparseMatrixCSC)
    if all(isprecise.(nodes))
        return BayesianNetwork(nodes, topology, A)
    else
        return CredalNetwork(nodes, topology, A)
    end
end

# Final step of `reduce`: pick the concrete network type for a (reduced) enhanced network. If any node is still continuous it cannot become a purely discrete network, so return the eBN unchanged; otherwise all nodes are discrete, giving a BayesianNetwork when every node is precise and a CredalNetwork when some are imprecise.
function _dispatch(ebn::EnhancedBayesianNetwork)
    if any(isa.(ebn.nodes, EnhancedBayesianNetworks.AbstractContinuousNode))
        return ebn
    end
    return _dispatch(Vector{DiscreteNode}(ebn.nodes), ebn.topology, ebn.A)
end
