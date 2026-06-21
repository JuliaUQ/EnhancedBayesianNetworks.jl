# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

# Internal base values — all user parameters scale these.
const _BASE_NODESIZE = 0.05   # circle radius / rect half-width (canvas units)
const _BASE_NODE_ASPECT = 0.6    # rect half-height = nodesize × aspect
const _BASE_ARROWLENGTH = 0.03   # arrowhead wing length (canvas units)
const _BASE_ARROWANGLE = π / 9  # arrowhead half-opening angle (fixed)
const _BASE_EDGEWIDTH = 0.3mm  # edge stroke width
const _BASE_LABELSIZE = 8pt    # node label font size
const _BASE_TITLESIZE = 10pt   # title font size
const _BORDER_PAD = 0.12   # fraction of canvas kept free at each edge

"""
    gplot(net::AbstractNetwork; kwargs...)

Plot an `AbstractNetwork` using pure Compose.jl.

**Layout**: hierarchical top-down. Root nodes (no parents) sit at the top;
every other node is placed one layer below its deepest parent. Nodes within
a layer are spread evenly across the width.

**Shapes**:
- `AbstractContinuousNode` → circle
- `AbstractDiscreteNode`   → rectangle

**Colours**:
- `FunctionalNode` (`ContinuousFunctionalNode` | `DiscreteFunctionalNode`) → orange
- Non-functional nodes (`ContinuousNode`, `DiscreteNode`) → green

**Keyword arguments**:

| Name            | Default | Description                                              |
|-----------------|---------|----------------------------------------------------------|
| `nodesize`      | `1.0`   | Scale factor for node size (circles and rectangles)      |
| `arrowsize`     | `1.0`   | Scale factor for arrowhead size                          |
| `edgesize`      | `1.0`   | Scale factor for edge stroke width                       |
| `nodelabelsize` | `1.0`   | Scale factor for node label font size                    |
| `title`         | `""`    | Optional title string drawn above the graph              |
| `title_size`    | `1.0`   | Scale factor for title font size                         |
"""
function gplot(net::EnhancedBayesianNetworks.AbstractNetwork;
    nodesize=1.0,
    arrowsize=1.0,
    edgesize=1.0,
    nodelabelsize=1.0,
    title="",
    title_size=1.0,
)
    node_list = net.nodes
    n = length(node_list)

    hw = _BASE_NODESIZE * nodesize
    hh = hw * _BASE_NODE_ASPECT
    al = _BASE_ARROWLENGTH * arrowsize
    ew = _BASE_EDGEWIDTH * edgesize
    ls = _BASE_LABELSIZE * nodelabelsize
    ts = _BASE_TITLESIZE * title_size

    # ── positions ────────────────────────────────────────────────────────────
    locs_x, locs_y = _layered_positions(net.A, _BORDER_PAD)

    # ── edges ────────────────────────────────────────────────────────────────
    edge_list = [(i, j) for i in 1:n for j in 1:n if net.A[i, j] != 0]
    edge_lines, edge_arrows = _build_edges(
        edge_list, locs_x, locs_y, node_list,
        hw, hh, al, _BASE_ARROWANGLE
    )

    # ── node shapes ──────────────────────────────────────────────────────────
    circle_ctxs, rect_ctxs = _build_node_contexts(locs_x, locs_y, node_list, hw, hh)

    # ── labels ───────────────────────────────────────────────────────────────
    label_ctx = compose(context(),
        Compose.text(locs_x, locs_y,
            string.(getfield.(node_list, :name)),
            [hcenter], [vcenter]),
        fill("black"),
        fontsize(ls),
        Compose.font("Helvetica")
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
    Compose.set_default_graphic_size(20cm, 20cm)

    legend = _build_legend()

    compose(context(),
        title_ctx,
        label_ctx,                                                        # labels (front)
        circle_ctxs...,                                                   # circular nodes
        rect_ctxs...,                                                     # rectangular nodes
        legend,
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

"""
    _compute_layers(A) -> Vector{Int}

Assigns a display layer to every node.
- Layer 0: root nodes (no incoming edges).
- Otherwise: `max(parent layers) + 1`.
Propagates iteratively until stable (correct for any DAG).
"""
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

"""
    _layered_positions(A, border_pad) -> locs_x, locs_y

Spreads nodes evenly within each layer.
`border_pad` (fraction of canvas) keeps nodes away from the canvas edge.
y = 0 is the top (roots), y = 1 is the bottom (leaves).
"""
function _layered_positions(A::SparseMatrixCSC, border_pad::Float64=0.12)
    n = size(A, 1)
    layers = _compute_layers(A)
    max_layer = maximum(layers)

    layer_groups = [Int[] for _ in 0:max_layer]
    for i in 1:n
        push!(layer_groups[layers[i]+1], i)
    end

    locs_x = zeros(n)
    locs_y = zeros(n)
    inner = 1.0 - 2 * border_pad   # usable canvas fraction

    for (l, group) in enumerate(layer_groups)
        isempty(group) && continue
        k = length(group)
        for (pos, idx) in enumerate(group)
            x_frac = k == 1 ? 0.5 : (pos - 1) / (k - 1)
            y_frac = (l - 1) / max(max_layer, 1)
            locs_x[idx] = border_pad + x_frac * inner
            locs_y[idx] = border_pad + y_frac * inner
        end
    end

    return locs_x, locs_y
end

# ─────────────────────────────────────────────────────────────────────────────
# Shape-aware border attachment points
# ─────────────────────────────────────────────────────────────────────────────

"""
    _circle_border(cx, cy, θ, r) -> (x, y)

Point on a circle's border in direction θ from its centre.
"""
function _circle_border(cx, cy, θ, r)
    return cx + r * cos(θ), cy + r * sin(θ)
end

"""
    _rect_border(cx, cy, θ, hw, hh) -> (x, y)

Point on an axis-aligned rectangle's border in direction θ from its centre.
Uses slab-intersection: finds the t at which the ray exits each half-slab
and takes the minimum (first exit face).
"""
function _rect_border(cx, cy, θ, hw, hh)
    dx, dy = cos(θ), sin(θ)
    tx = abs(dx) < 1e-12 ? Inf : hw / abs(dx)
    ty = abs(dy) < 1e-12 ? Inf : hh / abs(dy)
    t = min(tx, ty)
    return cx + t * dx, cy + t * dy
end

"""
    _border_point(node, cx, cy, θ, hw, hh) -> (x, y)

Dispatches to the correct border computation based on node shape:
- `AbstractContinuousNode` → circle  (radius = hw)
- `AbstractDiscreteNode`   → rectangle (half-width = hw, half-height = hh)
"""
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

"""
    _build_edges(edge_list, locs_x, locs_y, nodes, hw, hh, arrowlength, angleoffset)

For each directed edge (i → j):
1. Computes θ = angle from centre i to centre j.
2. Finds the departure point on node i's border (in direction θ).
3. Finds the arrival point on node j's border (in direction θ + π, i.e. opposite).
4. Places the arrowhead triangle with its tip exactly on node j's border.
5. Ends the line at the arrowhead base midpoint so it doesn't poke through.

Returns `(line_primitive, polygon_primitive)` ready for Compose.
"""
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
# Node colour and shape contexts
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

"""
    _node_strokewidth(node) -> Compose measure

Returns a thicker border for `AbstractContinuousNode`s whose `discretization`
field is non-empty, and the standard thin border for everything else.
"""
function _node_strokewidth(node::AbstractNode)
    if node isa AbstractContinuousNode && !isempty(node.discretization)
        return 1.2mm   # thick border: discretized continuous node
    else
        return 0.3mm   # standard border
    end
end

"""
    _build_node_contexts(locs_x, locs_y, node_list, hw, hh)

Returns two vectors of Compose contexts, one per node:
- Circles for `AbstractContinuousNode`
- Rectangles for `AbstractDiscreteNode`

Each context carries its own fill and stroke so mixed networks render correctly.
`AbstractContinuousNode`s with a non-empty `discretization` get a thicker border.
"""
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

# ─────────────────────────────────────────────────────────────────────────────
# Legend functionality
# ─────────────────────────────────────────────────────────────────────────────

function _build_legend()

    compose(
        context(0.8, 0.68, 0.18, 0.25),

        # border
        compose(
            context(),
            rectangle(),
            fill(nothing),
            stroke("black"),
            linewidth(0.3mm)
        ),

        # Precise
        compose(
            context(),
            circle(0.12, 0.12, 0.03),
            fill("lightgreen")
        ),

        # Imprecise
        compose(
            context(),
            circle(0.12, 0.27, 0.03),
            fill("limegreen")
        ),

        # Functional
        compose(
            context(),
            circle(0.12, 0.42, 0.03),
            fill("orange")
        ),

        # Continuous
        compose(
            context(),
            circle(0.12, 0.58, 0.03),
            fill(nothing),
            stroke("black")
        ),

        # Discrete
        compose(
            context(),
            rectangle(0.09, 0.71, 0.06, 0.06),
            fill(nothing),
            stroke("black")
        ),

        # Discretized continuous
        compose(
            context(),
            circle(0.12, 0.88, 0.03),
            fill("lightgreen"),
            stroke("black"),
            linewidth(1.2mm)
        ),

        # Labels
        compose(
            context(),
            text(0.22, 0.12, "Precise", hleft, vcenter)
        ), compose(
            context(),
            text(0.22, 0.27, "Imprecise", hleft, vcenter)
        ), compose(
            context(),
            text(0.22, 0.42, "Functional", hleft, vcenter)
        ), compose(
            context(),
            text(0.22, 0.58, "Continuous", hleft, vcenter)
        ), compose(
            context(),
            text(0.22, 0.74, "Discrete", hleft, vcenter)
        ), compose(
            context(),
            text(0.22, 0.88, "Discretized", hleft, vcenter)
        ), fontsize(10pt)
    )
end
