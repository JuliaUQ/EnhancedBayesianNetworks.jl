# Final step of `reduce`: pick the concrete network type for a (reduced) enhanced network. If any node
# is still continuous it cannot become a purely discrete network, so return the eBN unchanged; otherwise
# all nodes are discrete, giving a BayesianNetwork when every node is precise and a CredalNetwork when
# some are imprecise.
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

# Reached via `reduce`/`dispatch`: narrow a CredalNetwork to a BayesianNetwork when all its nodes turn out precise; otherwise keep it credal.
function dispatch(cn::CredalNetwork)
    if all(isprecise.(cn.nodes))
        return BayesianNetwork(Vector{DiscreteNode}(cn.nodes), cn.topology, cn.A)
    else
        return cn
    end
end