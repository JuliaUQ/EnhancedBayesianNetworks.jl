# Compact
function Base.show(io::IO, p::Posterior)
    q = join(string.(p.query), ", ")
    if isempty(p.evidence)
        print(io, "Posterior P(", q, ")")
    else
        ev = join(["$(n)=$(s)" for (n, s) in p.evidence], ", ")
        print(io, "Posterior P(", q, " | ", ev, ")")
    end
end

#REPL
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