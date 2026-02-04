## Floyd–Warshall Modification
function transitive_closure(A::SparseMatrixCSC{Bool,Int64})
    n = size(A, 1)
    R = copy(A)

    for k in 1:n
        for i in 1:n
            if R[i, k]
                R[i, :] .|= R[k, :]
            end
        end
    end

    return R
end

function iscyclic(A::SparseMatrixCSC{Bool,Int64})
    R = transitive_closure(A)
    H = R .& R'
    if nnz(H) != 0
        return true
    else
        return false
    end
end

function isconnected(A::SparseMatrixCSC{Bool,Int64})
    A_undirected = A .| A'
    R = transitive_closure(A_undirected)
    if all(R .| I(size(A, 1)))
        return true
    else
        return false
    end
end