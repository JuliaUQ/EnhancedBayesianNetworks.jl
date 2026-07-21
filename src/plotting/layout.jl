# ─────────────────────────────────────────────────────────────────────────────
# Layout: layered top-down positions from the adjacency matrix
# ─────────────────────────────────────────────────────────────────────────────
function _compute_layers(A::SparseMatrixCSC)
    n = size(A, 1)
    layer = fill(-1, n)
    for i in 1:n
        nnz(A[:, i]) == 0 && (layer[i] = 0)
    end
    changed = true
    while changed
        changed = false
        for j in 1:n
            parent_idxs = findnz(A[:, j])[1]
            isempty(parent_idxs) && continue
            all(layer[p] >= 0 for p in parent_idxs) || continue
            new_layer = maximum(layer[p] for p in parent_idxs) + 1
            if layer[j] != new_layer
                layer[j] = new_layer
                changed = true
            end
        end
    end
    return layer
end

function _layered_positions(A::SparseMatrixCSC, border_pad::Float64=0.12, top_pad::Float64=0.12)
    n = size(A, 1)
    layers = _compute_layers(A)
    max_layer = maximum(layers)

    layer_groups = [Int[] for _ in 0:max_layer]
    for i in 1:n
        push!(layer_groups[layers[i]+1], i)
    end

    locs_x = zeros(n)
    locs_y = zeros(n)
    innerx = 1 - 2border_pad
    innery = 1 - border_pad - top_pad

    for (l, group) in enumerate(layer_groups)
        isempty(group) && continue
        k = length(group)
        for (pos, idx) in enumerate(group)
            x_frac = k == 1 ? 0.5 : (pos - 1) / (k - 1)
            y_frac = (l - 1) / max(max_layer, 1)
            locs_x[idx] = border_pad + x_frac * innerx
            locs_y[idx] = top_pad + y_frac * innery
        end
    end

    return locs_x, locs_y
end
