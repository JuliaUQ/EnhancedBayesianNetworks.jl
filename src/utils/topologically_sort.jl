function topologically_sort(A::SparseMatrixCSC)
    indeg = vec(sum(A, dims=1))   # incoming edges
    queue = findall(==(0), indeg)
    order = Int[]
    while !isempty(queue)
        v = popfirst!(queue)
        push!(order, v)
        for w in findall(A[v, :])   # nodes v → w
            indeg[w] -= 1
            if indeg[w] == 0
                push!(queue, w)
            end
        end
    end
    return order
end