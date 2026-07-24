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

        parstr = _topology_parents(net, node)

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

        parstr = _topology_parents(bn, node)

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

        parstr = _topology_parents(cn, node)

        println(
            io,
            rpad(string(i), 4),
            rpad(string(node.name), 16),
            rpad(precision, 12),
            parstr
        )
    end
end