# Posterior: "P(query | evidence)" header, then the probability table row per state combination.
function Base.show(io::IO, p::Posterior)
    print(io, "Posterior ", _pq_string(p.query, p.evidence))
end
function Base.show(io::IO, ::MIME"text/plain", p::Posterior)
    f = p.factor
    ns = p.schema
    println(io, "Posterior ", _pq_string(p.query, p.evidence))
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
    print(io, "CredalPosterior ", _pq_string(p.query, p.evidence))
end
function Base.show(io::IO, ::MIME"text/plain", p::CredalPosterior)
    lower = p.lower
    upper = p.upper
    ns = p.schema
    println(io, "CredalPosterior ", _pq_string(p.query, p.evidence))
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