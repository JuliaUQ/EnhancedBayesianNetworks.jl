# verify that each child of a discrete node have the node name in its CPT
function verify_discrete(node::DiscreteNode, ch::AbstractVector{<:AbstractNode})
    not_functional_children = filter(x -> !isa(x, FunctionalNode), ch)
    for child in not_functional_children
        cols = Symbol.(Set(names(child.cpt.data)))
        if node.name ∉ cols
            error("Invalid Network: node $(child.name) does not have the node(s) $(node.name) in its CPT")
        end
    end
end

# verify that all the children of a continuous or functional nodes are only functional nodes
function verify_continuous_and_functional(node::Union{ContinuousNode,FunctionalNode}, ch::AbstractVector{<:AbstractNode})
    not_functional_children = filter(x -> !isa(x, FunctionalNode), ch)
    if !isempty(not_functional_children)
        error("Invalid Network: node(s) $([i.name for i in not_functional_children]) are not functional node(s) and cannot be children of the continuous/functional node $(node.name)")
    end
end