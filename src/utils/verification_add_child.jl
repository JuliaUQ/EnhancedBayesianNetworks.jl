## No recursion in BayesianNetworks
function _verify_no_recursion(par::AbstractNode, ch::AbstractNode)
    if par == ch
        error("Recursion on the same node '$(par.name)' is not allowed in EnhancedBayesianNetworks")
    end
end
## Root Nodes cannot be childrens
function _verify_root(_::AbstractNode, ch::AbstractNode)
    if isroot(ch)
        error("node '$(ch.name)' is a root node and cannot have parents")
    end
end
## Check parents is in the scenarios with all its states
function _verify_child(par::AbstractNode, ch::AbstractNode)
    if !isa(ch, FunctionalNode) && !isa(par, FunctionalNode)
        if string(par.name) ∉ names(ch.cpt.data)
            error("trying to set node '$(ch.name)' as child of node '$(par.name)', but '$(ch.name)' has a cpt that does not contains '$(par.name)' in the scenarios: $(ch.cpt)")
        end
        par_states = states(par)
        scenario2check = unique(ch.cpt.data[!, par.name])
        if any(par_states .∉ [scenario2check])
            error("child node '$(ch.name)' has scenarios $scenario2check, that is not coherent with its parent node '$(par.name)' with states $par_states")
        end
    end
end
## Check children of functional node needs to be functional Nodes
function _verify_functional_node(par::AbstractNode, ch::AbstractNode)
    if isa(par, FunctionalNode) && !isa(ch, FunctionalNode)
        error("functional node '$(par.name)' can have only functional children. '$(ch.name)' is not a functional node")
    end
end

## For BayesianNetwork2be
## No recursion in BayesianNetworks
function _verify_no_recursion(par::Symbol, ch::Symbol)
    if par == ch
        error("Recursion on the same node '$(par)' is not allowed in BayesianNetworks")
    end
end
