# Discrete Nodes
function Base.show(io::IO, node::DiscreteNode)
    print(io, "DiscreteNode(", node.name, ", parents=", parents(node), ", states=", states(node), ")")
end
function Base.show(io::IO, ::MIME"text/plain", node::DiscreteNode)
    println(io, "DiscreteNode: ", node.name)
    p = parents(node)
    if isempty(p)
        println(io, "Parents: none")
    else
        println(io, "Parents: ", join(string.(p), ", "))
    end
    println(io, "States: ", join(string.(states(node)), ", "))
    if isprecise(node)
        println(io, "Type: Precise")
    else
        println(io, "Type: Credal")
    end
    if !isempty(node.parameters)
        println(io)
        println(io, "Parameters:")
        for (name, pars) in node.parameters
            println(io, "  ", name, ": ", join(string.(pars), ", "))
        end
    end
    println(io)
    show(io, MIME"text/plain"(), node.cpt.data)
end

# Continuous Nodes
function Base.show(io::IO, node::ContinuousNode)
    print(io, "ContinuousNode(", node.name, ", parents=", parents(node), ", discretization=", typeof(node.discretization).name.name, ")"
    )
end
function Base.show(io::IO, ::MIME"text/plain", node::ContinuousNode)
    println(io, "ContinuousNode: ", node.name)
    p = parents(node)
    if isempty(p)
        println(io, "Parents: none")
    else
        println(io, "Parents: ", join(string.(p), ", "))
    end

    println(io, "Discretization: ", nameof(typeof(node.discretization))
    )
    println(io, "Type: ", isprecise(node) ? "Precise" : "Imprecise")
    try
        bounds = _distribution_bounds(node)
        println(io, "Support: [", bounds[1], ", ", bounds[2], "]")
    catch
    end
    println(io)
    show(io, MIME"text/plain"(), node.cpt.data)
end
# Functional Nodes
function Base.show(io::IO, node::ContinuousFunctionalNode)
    print(
        io, "ContinuousFunctionalNode(", node.name, ", models=", length(node.models), ", nbins=", node.nbins, ")")
end
function Base.show(io::IO, ::MIME"text/plain", node::ContinuousFunctionalNode)
    println(io, "ContinuousFunctionalNode: ", node.name)
    println(io, "Models: ", length(node.models))
    println(io, "Discretization: ", nameof(typeof(node.discretization))
    )
    println(io, "Bins: ", node.nbins)
    println(io, "Simulation: ", nameof(typeof(node.simulation)))
    if node.simulation isa ScenariosTable
        println(io)
        show(io, MIME"text/plain"(), node.simulation.data)
    end
end

function Base.show(io::IO, node::DiscreteFunctionalNode)
    print(io, "DiscreteFunctionalNode(", node.name, ", states=", states(node), ", models=", length(node.models), ")")
end
function Base.show(io::IO, ::MIME"text/plain", node::DiscreteFunctionalNode)
    println(io, "DiscreteFunctionalNode: ", node.name)
    println(io, "States: ", join(string.(states(node)), ", "))
    println(io, "Models: ", length(node.models))
    println(io, "Simulation: ", nameof(typeof(node.simulation)))
    println(io, "Parameters: ", length(node.parameters)
    )
    if !isempty(node.parameters)
        println(io)
        for (name, pars) in node.parameters
            println(io, "  ", name, ": ", length(pars), " parameter(s)")
        end
    end
    if node.simulation isa ScenariosTable
        println(io)
        show(io, MIME"text/plain"(), node.simulation.data)
    end
end

# Posterior
function Base.show(io::IO, p::Posterior)
    q = join(string.(p.query), ", ")
    if isempty(p.evidence)
        print(io, "Posterior P(", q, ")")
    else
        ev = join(["$(n)=$(s)" for (n, s) in p.evidence], ", ")
        print(io, "Posterior P(", q, " | ", ev, ")")
    end
end
function Base.show(io::IO, ::MIME"text/plain", p::Posterior)
    f = p.factor
    ns = p.schema
    q = join(string.(p.query), ", ")
    if isempty(p.evidence)
        println(io, "Posterior P(", q, ")")
    else
        ev = join(["$(n)=$(s)" for (n, s) in p.evidence], ", ")
        println(io, "Posterior P(", q, " | ", ev, ")")
    end
    println(io)
    names = ns.idx_to_node[f.vars]
    for name in names
        print(io, name, "\t")
    end
    println(io, "Probability")
    println(io, repeat("-", 12 * (length(names) + 1)))

    for I in CartesianIndices(f.table)
        idxs = Tuple(I)
        for (var, idx) in zip(f.vars, idxs)
            print(io, ns.idx_to_state[var][idx], "\t")
        end
        println(io, f.table[I])
    end
end

# CredalPosterior
function Base.show(io::IO, p::CredalPosterior)
    q = join(string.(p.query), ", ")
    if isempty(p.evidence)
        print(io, "CredalPosterior P(", q, ")")
    else
        ev = join(["$(n)=$(s)" for (n, s) in p.evidence], ", ")
        print(io, "CredalPosterior P(", q, " | ", ev, ")")
    end
end
function Base.show(io::IO, ::MIME"text/plain", p::CredalPosterior)
    lower = p.lower
    upper = p.upper
    ns = p.schema
    q = join(string.(p.query), ", ")
    if isempty(p.evidence)
        println(io, "CredalPosterior P(", q, ")")
    else
        ev = join(["$(n)=$(s)" for (n, s) in p.evidence], ", ")
        println(io, "CredalPosterior P(", q, " | ", ev, ")")
    end
    println(io)
    names = ns.idx_to_node[lower.vars]
    for name in names
        print(io, name, "\t")
    end
    println(io, "Interval")
    println(io, repeat("-", 12 * (length(names) + 1)))
    for I in CartesianIndices(lower.table)
        idxs = Tuple(I)
        for (var, idx) in zip(lower.vars, idxs)
            print(io, ns.idx_to_state[var][idx], "\t")
        end
        l = round(lower.table[I], sigdigits=6)
        u = round(upper.table[I], sigdigits=6)
        println(io, "[", l, ", ", u, "]")
    end
    println(io)
    println(io, "Extreme posteriors: ", length(p.posteriors))
end