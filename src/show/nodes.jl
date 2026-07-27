# Discrete nodes: header line + parents/states/precision, then the CPT and any parameters.
function Base.show(io::IO, node::DiscreteNode)
    print(io, "DiscreteNode(", node.name, ", parents=", parents(node), ", states=", states(node), ")")
end
function Base.show(io::IO, ::MIME"text/plain", node::DiscreteNode)
    println(io, "DiscreteNode: ", node.name)
    _show_parents(io, node)
    println(io, "States: ", join(string.(states(node)), ", "))
    if isprecise(node)
        println(io, "Type: Precise")
    else
        println(io, "Type: Credal")
    end
    _show_parameters(io, node.parameters)
    println(io)
    show(io, MIME"text/plain"(), node.cpt.data)
end

# Continuous nodes: header + parents/discretization/precision/support, then the CPT.
function Base.show(io::IO, node::ContinuousNode)
    print(io, "ContinuousNode(", node.name, ", parents=", parents(node), ", discretization=", typeof(node.discretization).name.name, ")"
    )
end
function Base.show(io::IO, ::MIME"text/plain", node::ContinuousNode)
    println(io, "ContinuousNode: ", node.name)
    _show_parents(io, node)
    _show_discretization(io, node.discretization)
    println(io, "Type: ", isprecise(node) ? "Precise" : "Imprecise")
    # Support line is display-only; swallow the error when bounds can't be computed for this node.
    try
        bounds = _distribution_bounds(node)
        println(io, "Support: [", bounds[1], ", ", bounds[2], "]")
    catch
    end
    println(io)
    show(io, MIME"text/plain"(), node.cpt.data)
end

# Functional nodes: header + models/discretization/simulation (+ parameters for discrete), then the per-scenario simulation table.
function Base.show(io::IO, node::ContinuousFunctionalNode)
    print(
        io, "ContinuousFunctionalNode(", node.name, ", models=", length(node.models), ", nbins=", node.nbins, ")")
end
function Base.show(io::IO, ::MIME"text/plain", node::ContinuousFunctionalNode)
    println(io, "ContinuousFunctionalNode: ", node.name)
    _show_models(io, node.models)
    _show_discretization(io, node.discretization)
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
    _show_models(io, node.models)
    println(io, "Simulation: ", nameof(typeof(node.simulation)))
    _show_parameters(io, node.parameters)
    if node.simulation isa ScenariosTable
        println(io)
        show(io, MIME"text/plain"(), node.simulation.data)
    end
end