function dispatch(ebn::EnhancedBayesianNetwork)
    if any(isa.(ebn.nodes, EnhancedBayesianNetworks.AbstractContinuousNode))
        return ebn
    else
        nodes = Vector{DiscreteNode}(ebn.nodes)
        if all(isprecise.(ebn.nodes))
            net = BayesianNetwork(nodes)
        else
            net = CredalNetwork(nodes)
        end
        net.A = ebn.A
        net.topology = ebn.topology
        return net
    end
end

function dispatch(cn::CredalNetwork)
    if all(isprecise.(cn.nodes))
        bn = BayesianNetwork(Vector{DiscreteNode}(cn.nodes))
        bn.A = cn.A
        bn.topology = cn.topology
        return bn
    else
        return cn
    end
end