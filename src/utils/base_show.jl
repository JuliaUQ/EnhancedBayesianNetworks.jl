# Pretty-printing for the library's public types. Each type gets two methods: a compact one-line
# `show(io, x)` (used in arrays, REPL echo of a field) and a full `show(io, ::MIME"text/plain", x)`
# (used when the object is displayed on its own) that lays out its structure and underlying table.

# Discrete nodes: header line + parents/states/precision, then the CPT and any parameters.
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
        println(io, "Parameters:")
        for (name, pars) in node.parameters
            println(io, "  ", name, ": ", join(string.(pars), ", "))
        end
    end
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
    p = parents(node)
    if isempty(p)
        println(io, "Parents: none")
    else
        println(io, "Parents: ", join(string.(p), ", "))
    end

    d = node.discretization
    if !isempty(d)
        println(io, "Discretization: ", nameof(typeof(d)))
        if d isa ApproximatedDiscretization
            println(io, "  Sigma: ", d.sigma)
        end
        println(io, "  Intervals: ", join(d.intervals, ", "))
    end

    println(io, "Type: ", isprecise(node) ? "Precise" : "Imprecise")
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
    if !isempty(node.models)
        println(io, "Models: ", length(node.models))
        println(io, "  Names: ", join(getproperty.(node.models, :name), ", "))
    end
    d = node.discretization
    if !isempty(d)
        println(io, "Discretization: ", nameof(typeof(d)))
        if d isa ApproximatedDiscretization
            println(io, "  Sigma: ", d.sigma)
        end
        println(io, "  Intervals: ", join(d.intervals, ", "))
    end

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

    if !isempty(node.models)
        println(io, "Models: ", length(node.models))
        println(io, "  Names: ", join(getproperty.(node.models, :name), ", "))
    end

    println(io, "Simulation: ", nameof(typeof(node.simulation)))
    if !isempty(node.parameters)
        println(io, "Parameters:")
        for (name, pars) in node.parameters
            println(io, "  ", name, ": ", join(string.(pars), ", "))
        end
    end

    if node.simulation isa ScenariosTable
        println(io)
        show(io, MIME"text/plain"(), node.simulation.data)
    end
end

# Posterior: "P(query | evidence)" header, then the probability table row per state combination.
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

# CredalPosterior: like Posterior but printing [lower, upper] bounds, plus the count of extreme posteriors.
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

# EnhancedBayesianNetwork: counts by node kind, then a topology table (node, type, precision, parents).
function Base.show(io::IO, net::EnhancedBayesianNetwork)
    print(io, "EnhancedBayesianNetwork(", length(net.nodes), " nodes)")
end
function Base.show(io::IO, ::MIME"text/plain", net::EnhancedBayesianNetwork)

    println(io, "EnhancedBayesianNetwork")
    println(io)

    println(io, "Nodes: ", length(net.nodes))
    println(io, "Edges: ", count(net.A))

    nd = count(n -> n isa AbstractDiscreteNode, net.nodes)
    nc = count(n -> n isa AbstractContinuousNode, net.nodes)
    nf = count(n -> n isa FunctionalNode, net.nodes)

    println(io, "Discrete nodes: ", nd)
    println(io, "Continuous nodes: ", nc)
    println(io, "Functional nodes: ", nf)

    println(io)
    println(io, "Topology:")
    println(io)

    println(
        io,
        rpad("Node", 20),
        rpad("Type", 24),
        rpad("Precision", 12),
        "Parents"
    )

    println(io, repeat("-", 80))

    for node in net.nodes

        typ =
            node isa DiscreteNode ? "Discrete" :
            node isa ContinuousNode ? "Continuous" :
            node isa DiscreteFunctionalNode ? "DiscreteFunctional" :
            node isa ContinuousFunctionalNode ? "ContinuousFunctional" :
            string(nameof(typeof(node)))

        precision =
            node isa FunctionalNode ? "" :
            node isa AbstractDiscreteNode ?
            (isprecise(node) ? "Precise" : "Credal") :
            (isprecise(node) ? "Precise" : "Imprecise")

        parstr =
            isempty(parents(net, node)) ?
            "-" :
            join(string.(parents(net, node)), ", ")

        println(io, rpad(string(node.name), 20), rpad(typ, 24), rpad(precision, 12), parstr)
    end
end

# BayesianNetwork: node/edge counts, then a topology table (index, node, states, parents).
function Base.show(io::IO, bn::BayesianNetwork)
    print(io, "BayesianNetwork(", length(bn.nodes), " nodes)")
end

function Base.show(io::IO, ::MIME"text/plain", bn::BayesianNetwork)

    println(io, "BayesianNetwork")
    println(io)

    println(io, "Nodes: ", length(bn.nodes))
    println(io, "Edges: ", count(bn.A))

    println(io)
    println(io, "Topology:")
    println(io)

    println(
        io,
        rpad("#", 4),
        rpad("Node", 16),
        rpad("States", 20),
        "Parents"
    )

    println(io, repeat("-", 80))

    for (i, node) in enumerate(bn.nodes)

        statestr = join(string.(states(node)), ", ")

        parstr =
            isempty(parents(bn, node)) ?
            "-" :
            join(string.(parents(bn, node)), ", ")

        println(
            io,
            rpad(string(i), 4),
            rpad(string(node.name), 16),
            rpad(statestr, 20),
            parstr
        )
    end
end

# CredalNetwork: precise/credal counts, then a topology table (index, node, precision, parents).
function Base.show(io::IO, cn::CredalNetwork)
    print(io, "CredalNetwork(", length(cn.nodes), " nodes)")
end

function Base.show(io::IO, ::MIME"text/plain", cn::CredalNetwork)

    println(io, "CredalNetwork")
    println(io)

    println(io, "Nodes: ", length(cn.nodes))
    println(io, "Edges: ", count(cn.A))

    nprecise = count(isprecise, cn.nodes)
    nimprecise = length(cn.nodes) - nprecise

    println(io, "Precise nodes: ", nprecise)
    println(io, "Credal nodes: ", nimprecise)

    println(io)
    println(io, "Topology:")
    println(io)

    println(
        io,
        rpad("#", 4),
        rpad("Node", 16),
        rpad("Precision", 12),
        "Parents"
    )

    println(io, repeat("-", 80))

    for (i, node) in enumerate(cn.nodes)

        precision =
            isprecise(node) ? "Precise" : "Credal"

        parstr =
            isempty(parents(cn, node)) ?
            "-" :
            join(string.(parents(cn, node)), ", ")

        println(
            io,
            rpad(string(i), 4),
            rpad(string(node.name), 16),
            rpad(precision, 12),
            parstr
        )
    end
end