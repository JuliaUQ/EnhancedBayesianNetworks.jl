# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Internal base values — all user parameters scale these.
const _BASE_LABELSIZE = 8pt    # node label font size
const _BASE_TITLESIZE = 18pt   # title font size
const _BORDER_PAD = 0.12   # fraction of canvas kept free at each edge


function gplot(net::EnhancedBayesianNetworks.AbstractNetwork;
    node_scale::Float64=1.0,
    label_scale::Float64=1.0,
    title::String="",
    title_scale::Float64=1.0,
    figsize::Tuple=(20cm, 20cm),
    legend::Bool=false,
    legend_scale::Float64=1.0
)
    node_list = net.nodes
    n = length(node_list)

    hw = node_scale * 0.05
    hh = hw * 0.6
    al = 0.03 * node_scale

    ew = 0.3mm * node_scale
    ls = _BASE_LABELSIZE * label_scale
    ts = _BASE_TITLESIZE * title_scale

    # ── positions ────────────────────────────────────────────────────────────
    top_pad = isempty(title) ? 0.12 : 0.18
    locs_x, locs_y = _layered_positions(net.A, _BORDER_PAD, top_pad)

    # ── edges ────────────────────────────────────────────────────────────────
    edge_list = [(i, j) for i in 1:n for j in 1:n if net.A[i, j] != 0]
    edge_lines, edge_arrows = _build_edges(
        edge_list, locs_x, locs_y, node_list,
        hw, hh, al, π / 9
    )

    # ── node shapes ──────────────────────────────────────────────────────────
    circle_ctxs, rect_ctxs = _build_node_contexts(locs_x, locs_y, node_list, hw, hh)

    # ── labels ───────────────────────────────────────────────────────────────
    label_ctxs = _build_labels(
        node_list,
        locs_x,
        locs_y,
        ls
    )

    # ── optional title ───────────────────────────────────────────────────────
    title_ctx = isempty(title) ? context() :
                compose(context(),
        Compose.text(0.5, _BORDER_PAD / 2, title, hcenter, vcenter),
        fill("black"),
        fontsize(ts),
        Compose.font("Helvetica")
    )

    # ── assemble (painter's order: back → front) ─────────────────────────────
    Compose.set_default_graphic_size(figsize[1], figsize[2])

    legend_ctx = legend ? _build_legend(legend_scale) : context()

    compose(context(),
        title_ctx,
        label_ctxs...,                                                        # labels (front)
        circle_ctxs...,                                                   # circular nodes
        rect_ctxs...,                                                     # rectangular nodes
        legend_ctx,
        compose(context(), edge_arrows, fill("black")),                   # arrowheads
        compose(context(), edge_lines, Compose.stroke("black"), linewidth(ew)), # edge lines (back)
    )
end

"""
    saveplot(p, filename::String)

Save a `gplot` result to an SVG file.
"""
function saveplot(p, filename::String)
    draw(SVG(filename), p)
end


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

# ─────────────────────────────────────────────────────────────────────────────
# Shape-aware border attachment points
# ─────────────────────────────────────────────────────────────────────────────

function _circle_border(cx, cy, θ, r)
    return cx + r * cos(θ), cy + r * sin(θ)
end

function _rect_border(cx, cy, θ, hw, hh)
    dx, dy = cos(θ), sin(θ)
    tx = abs(dx) < 1e-12 ? Inf : hw / abs(dx)
    ty = abs(dy) < 1e-12 ? Inf : hh / abs(dy)
    t = min(tx, ty)
    return cx + t * dx, cy + t * dy
end

function _border_point(node::AbstractNode, cx, cy, θ, hw, hh)
    if node isa AbstractContinuousNode
        return _circle_border(cx, cy, θ, hw)
    else  # AbstractDiscreteNode
        return _rect_border(cx, cy, θ, hw, hh)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Arrow geometry
# ─────────────────────────────────────────────────────────────────────────────

function _arrowcoords(θ, tip_x, tip_y, arrowlength, angleoffset)
    arr1 = (tip_x - arrowlength * cos(θ + angleoffset),
        tip_y - arrowlength * sin(θ + angleoffset))
    arr2 = (tip_x - arrowlength * cos(θ - angleoffset),
        tip_y - arrowlength * sin(θ - angleoffset))
    return arr1, arr2
end

function _midpoint(a, b)
    return (a[1] + b[1]) / 2, (a[2] + b[2]) / 2
end

# ─────────────────────────────────────────────────────────────────────────────
# Build all edges (lines + arrowhead polygons)
# ─────────────────────────────────────────────────────────────────────────────

function _build_edges(edge_list, locs_x, locs_y, nodes, hw, hh,
    arrowlength, angleoffset)
    isempty(edge_list) && return line([]), polygon([])

    lines = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(edge_list))
    arrows = Vector{Vector{Tuple{Float64,Float64}}}(undef, length(edge_list))

    for (e_idx, (i, j)) in enumerate(edge_list)
        Δx = locs_x[j] - locs_x[i]
        Δy = locs_y[j] - locs_y[i]
        θ = atan(Δy, Δx)

        # departure: border of source node i, shooting toward j
        startx, starty = _border_point(nodes[i], locs_x[i], locs_y[i], θ, hw, hh)
        # arrival:   border of target node j, shooting back toward i
        tip_x, tip_y = _border_point(nodes[j], locs_x[j], locs_y[j], θ + π, hw, hh)

        # arrowhead: tip sits exactly on j's border; base goes outward (away from j)
        arr1, arr2 = _arrowcoords(θ, tip_x, tip_y, arrowlength, angleoffset)
        base_mid = _midpoint(arr1, arr2)

        # line stops at arrowhead base so it doesn't show through the filled triangle
        lines[e_idx] = [(startx, starty), base_mid]
        arrows[e_idx] = [arr1, (tip_x, tip_y), arr2]
    end

    return line(lines), polygon(arrows)
end

# ─────────────────────────────────────────────────────────────────────────────
# Node colour, shape and label contexts
# ─────────────────────────────────────────────────────────────────────────────
function _node_color(node::AbstractNode)
    if node isa ContinuousNode
        if isprecise(node)
            return "lightgreen"  # pale orange
        else
            return "limegreen"     # bright orange
        end
    elseif node isa DiscreteNode
        if isprecise(node)
            return "lightgreen" # pale green
        else
            return "limegreen"  # bright green
        end
    else
        return "orange"  # fallback for other node types
    end
end

function _node_strokewidth(node::AbstractNode)
    if node isa AbstractContinuousNode && !isempty(node.discretization)
        return 1.2mm   # thick border: discretized continuous node
    else
        return 0.3mm   # standard border
    end
end

function _build_node_contexts(locs_x, locs_y, node_list, hw, hh)
    circle_ctxs = Compose.Context[]
    rect_ctxs = Compose.Context[]

    for (i, node) in enumerate(node_list)
        x, y = locs_x[i], locs_y[i]
        col = _node_color(node)
        lw = _node_strokewidth(node)

        if node isa AbstractContinuousNode
            push!(circle_ctxs,
                compose(context(),
                    circle(x, y, hw),
                    fill(col),
                    Compose.stroke("black"),
                    linewidth(lw)
                )
            )
        else  # AbstractDiscreteNode
            push!(rect_ctxs,
                compose(context(),
                    rectangle(x - hw, y - hh, 2hw, 2hh),
                    fill(col),
                    Compose.stroke("black"),
                    linewidth(lw)
                )
            )
        end
    end

    return circle_ctxs, rect_ctxs
end

function _build_labels(node_list, locs_x, locs_y, labelsize)
    labels = Compose.Context[]
    for (i, node) in enumerate(node_list)
        x = locs_x[i]
        y = locs_y[i]
        push!(
            labels,
            compose(context(), text(x, y - 0.01, string(node.name), hcenter, vcenter), fontsize(labelsize)))
        if node isa AbstractDiscreteNode
            push!(
                labels,
                compose(
                    context(),
                    text(
                        x,
                        y + 0.025,
                        "["*string(length(states(node)))*"]",
                        hcenter,
                        vcenter
                    ),
                    fontsize(0.8labelsize)
                )
            )
        end
    end
    return labels
end

# ─────────────────────────────────────────────────────────────────────────────
# Legend functionality
# ─────────────────────────────────────────────────────────────────────────────

function _build_legend(scale)

    r = 0.05 * scale
    rect_w = 0.09 * scale
    rect_h = 0.06 * scale

    bar_w = 0.09 * scale
    bar_h = 0.015 * scale

    fs = 10 * scale
    header_fs = 11 * scale

    compose(
        context(0.8, 0.65, 0.18, 0.3),

        # Border
        compose(
            context(),
            rectangle(),
            fill(nothing),
            stroke("black"),
            linewidth(0.3mm)
        ),

        ####################################################################
        # Shape section
        ####################################################################

        compose(
            context(),
            text(0.08, 0.06, "Shape", hleft, vcenter),
            fontsize(header_fs * pt)
        ),
        compose(
            context(),
            text(0.08, 0.54, "Color", hleft, vcenter),
            fontsize(header_fs * pt)
        ),

        # Continuous
        compose(
            context(),
            circle(0.12, 0.16, r),
            fill(nothing),
            stroke("black")
        ),
        compose(
            context(),
            text(0.22, 0.16, "Continuous", hleft, vcenter),
            fontsize(fs * pt)
        ),

        # Discrete
        compose(
            context(),
            rectangle(
                0.12 - rect_w/2,
                0.28 - rect_h/2,
                rect_w,
                rect_h
            ),
            fill(nothing),
            stroke("black")
        ),
        compose(
            context(),
            text(0.22, 0.28, "Discrete", hleft, vcenter),
            fontsize(fs * pt)
        ),

        # Discretized
        compose(
            context(),
            circle(0.12, 0.40, r),
            fill("lightgreen"),
            stroke("black"),
            linewidth(1.2mm * scale)
        ),
        compose(
            context(),
            text(0.22, 0.40, "Discretized", hleft, vcenter),
            fontsize(fs * pt)
        ),

        ####################################################################
        # Color section
        ####################################################################

        # Precise
        compose(
            context(),
            rectangle(
                0.125 - bar_w/2,
                0.635 - bar_h/2,
                bar_w,
                bar_h
            ),
            fill("lightgreen")
        ),
        compose(
            context(),
            text(0.22, 0.635, "Precise", hleft, vcenter),
            fontsize(fs * pt)
        ),

        # Imprecise
        compose(
            context(),
            rectangle(
                0.125 - bar_w/2,
                0.755 - bar_h/2,
                bar_w,
                bar_h
            ),
            fill("limegreen")
        ),
        compose(
            context(),
            text(0.22, 0.755, "Imprecise", hleft, vcenter),
            fontsize(fs * pt)
        ),

        # Functional
        compose(
            context(),
            rectangle(
                0.125 - bar_w/2,
                0.875 - bar_h/2,
                bar_w,
                bar_h
            ),
            fill("orange")
        ),
        compose(
            context(),
            text(0.22, 0.875, "Functional", hleft, vcenter),
            fontsize(fs * pt)
        )
    )
end