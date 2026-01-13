function _verify_evidence(a::Evidence, bn::BayesianNetwork)
    any(keys(a) .∉ [[i.name for i in bn.nodes]]) && error("all nodes in the $a have to be in the network")
    for (n, s) in a
        s ∉ states(bn.nodes[bn.topology_dict[n]]) && error("node states in $a must be coherent with the one defined in the network")
    end
end