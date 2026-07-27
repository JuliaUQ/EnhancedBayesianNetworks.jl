# Pretty-printing for the library's public types. 
# Each type gets two methods: 
# a compact one-line `show(io, x)` (used in arrays, REPL echo of a field) and 
# a full `show(io, ::MIME"text/plain", x)`(used when the object is displayed on its own) that lays out its structure and underlying table.

# "Parents: none" or "Parents: a, b" for a node's own parents.
function _show_parents(io, node)
    p = parents(node)
    if isempty(p)
        println(io, "Parents: none")
    else
        println(io, "Parents: ", join(string.(p), ", "))
    end
end

# Discretization block: scheme name, the sigma line only for ApproximatedDiscretization, then intervals.
# Prints nothing for an empty discretization.
function _show_discretization(io, d)
    if !isempty(d)
        println(io, "Discretization: ", nameof(typeof(d)))
        if d isa ApproximatedDiscretization
            println(io, "  Sigma: ", d.sigma)
        end
        println(io, "  Intervals: ", join(d.intervals, ", "))
    end
end

# "Parameters:" block, one indented line per state's parameter vector. Nothing when there are none.
function _show_parameters(io, parameters)
    if !isempty(parameters)
        println(io, "Parameters:")
        for (name, pars) in parameters
            println(io, "  ", name, ": ", join(string.(pars), ", "))
        end
    end
end

# "Models: N" + the indented list of model names. Nothing when there are no models.
function _show_models(io, models)
    if !isempty(models)
        println(io, "Models: ", length(models))
        println(io, "  Names: ", join(getproperty.(models, :name), ", "))
    end
end

# "P(query)" or "P(query | ev)" string shared by Posterior/CredalPosterior (both compact and plain).
function _pq_string(query, evidence)
    q = join(string.(query), ", ")
    if isempty(evidence)
        return string("P(", q, ")")
    else
        ev = join(["$(n)=$(s)" for (n, s) in evidence], ", ")
        return string("P(", q, " | ", ev, ")")
    end
end

# "-" or the comma-joined parent names, for a node's row in a network topology table.
_topology_parents(net, node) = isempty(parents(net, node)) ? "-" : join(string.(parents(net, node)), ", ")

include("nodes.jl")
include("posteriors.jl")
include("networks.jl")