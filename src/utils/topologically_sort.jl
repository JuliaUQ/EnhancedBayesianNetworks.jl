# Kahn's algorithm (A. B. Kahn, CACM 1962): repeatedly remove in-degree-0 vertices. A cyclic graph yields length(order) < size(A, 1).
function topologically_sort(A::SparseMatrixCSC)
    # in-degree of each vertex (column sums)
    indeg = vec(sum(A, dims=1))
    # sources: in-degree 0
    queue = findall(==(0), indeg)
    order = Int[]
    while !isempty(queue)
        v = popfirst!(queue)
        push!(order, v)
        # successors of v
        for w in findall(A[v, :])
            indeg[w] -= 1
            if indeg[w] == 0
                push!(queue, w)
            end
        end
    end
    return order
end