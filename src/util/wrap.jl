wrap(x) = [x]
wrap(x::AbstractArray) = x
wrap(x::Parameter) = [x]

flat(x) = [x, x]
flat(x::Interval) = [x.lb, x.ub]

function _by_row(evidence::Dict{Symbol,Symbol})
    k = collect(keys(evidence))
    v = collect(values(evidence))
    return map((n, s) -> n => ByRow(x -> x == s), k, v)
end