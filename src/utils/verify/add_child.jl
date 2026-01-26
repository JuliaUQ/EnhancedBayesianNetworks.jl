function verify_no_recursion(par::AbstractVector{<:AbstractNode}, ch::AbstractVector{<:AbstractNode})
    overlap = intersect(par, ch)
    if !isempty(overlap)
        error("Invalid eBN: node(s) '$([i.name for i in overlap])' have recursion")
    end
end

function verify_discrete(node::DiscreteNode, ch::AbstractVector{<:AbstractNode})
    for child in ch
        cols = Symbol.(Set(names(child.cpt.data)))
        if node.name ∉ cols
            error("Invalid eBN: node $(child.name) does not have the node(s) $(node.name) in its CPT")
        end
    end
end

function verify_continuous_and_functional(node::Union{ContinuousNode,FunctionalNode}, ch::AbstractVector{<:AbstractNode})
    not_functional_ch = filter(x -> !isa(x, FunctionalNode), ch)
    if !isempty(not_functional_ch)
        error("Invalid eBN: node(s) $([i.name for i in not_functional_ch]) are not functional node(s) and cannot be children of the continuous/functional node $(node.name)")
    end
end