function dispatch(ebn::EnhancedBayesianNetwork)
    if any(isa.(ebn.nodes, EnhancedBayesianNetworks.AbstractContinuousNode))
        return ebn
    else
        nodes = Vector{DiscreteNode}(ebn.nodes)
        if all(isprecise.(ebn.nodes))
            return BayesianNetwork(nodes, ebn.topology, ebn.A)
        else
            return CredalNetwork(nodes, ebn.topology, ebn.A)
        end
    end
end

function dispatch(cn::CredalNetwork)
    if all(isprecise.(cn.nodes))
        return BayesianNetwork(Vector{DiscreteNode}(cn.nodes), cn.topology, cn.A)
    else
        return cn
    end
end