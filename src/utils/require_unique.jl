# Return all elements that appear more than once in a vector (each duplicate listed once).
function _not_unique_elements(v::AbstractVector)
    seen = Set{eltype(v)}()
    dups = Set{eltype(v)}()

    for x in v
        if x in seen
            push!(dups, x)
        else
            push!(seen, x)
        end
    end
    return collect(dups)
end