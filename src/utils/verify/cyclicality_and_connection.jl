# Transitive closure via Floyd–Warshall: for each intermediate vertex k, every vertex reaching k inherits everything k reaches. Dense, since the closure is.
function transitive_closure(A::SparseMatrixCSC{Bool,Int64})
    n = size(A, 1)
    R = Matrix(A)
    for k in 1:n
        for i in 1:n
            if R[i, k]
                @views R[i, :] .|= R[k, :]
            end
        end
    end
    return R
end

# Cyclic if Kahn's sort can't order every vertex (length(order) < n).
iscyclic(A::SparseMatrixCSC{Bool,Int64}) = length(topologically_sort(A)) != size(A, 1)

# function iscyclic(A::SparseMatrixCSC{Bool,Int64})
#     R = transitive_closure(A)
#     H = R .& R'
#     if nnz(H) != 0
#         return true
#     else
#         return false
#     end
# end

# Undirected connectivity: symmetrise, take reachability, require every vertex pair connected (I covers the diagonal / self-reachability).
function isconnected(A::SparseMatrixCSC{Bool,Int64})
    A_undirected = A .| A'
    R = transitive_closure(A_undirected)
    return all(R .| I(size(A, 1)))
end