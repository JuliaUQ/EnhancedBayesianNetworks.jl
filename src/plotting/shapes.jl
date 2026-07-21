# Corner radius of rounded hexagons, as a fraction of the node half-width.
const _HEX_CORNER_RADIUS = 0.40

# Drawing shape of a node. Functional nodes are hexagons — rounded corners when
# continuous, pointy corners when discrete — plain nodes keep circle/rectangle.
# The functional types must be tested first: they are subtypes of the abstract
# continuous/discrete types.
function _node_shape(node::AbstractNode)
    if node isa ContinuousFunctionalNode
        return :rounded_hexagon
    elseif node isa DiscreteFunctionalNode
        return :hexagon
    elseif node isa ContinuousNode
        return :circle
    elseif node isa DiscreteNode
        return :rectangle
    end
end

# Flat-top hexagon inscribed in the (2hw × 2hh) box shared with rectangular nodes.
function _hexagon_vertices(cx, cy, hw, hh)
    return [(cx + hw, cy),
        (cx + hw / 2, cy + hh),
        (cx - hw / 2, cy + hh),
        (cx - hw, cy),
        (cx - hw / 2, cy - hh),
        (cx + hw / 2, cy - hh)]
end

# Outline of `verts` with every corner replaced by a circular arc of radius `r`,
# sampled into `arcpoints` segments so it can be drawn as a plain polygon —
# Compose has no rounded-polygon primitive.
function _rounded_polygon(verts, r::Float64, arcpoints::Int=8)
    n = length(verts)
    pts = Tuple{Float64,Float64}[]
    for k in 1:n
        cx, cy = verts[k]
        px, py = verts[mod1(k - 1, n)]
        nx, ny = verts[mod1(k + 1, n)]
        ux, uy = px - cx, py - cy
        vx, vy = nx - cx, ny - cy
        lu, lv = hypot(ux, uy), hypot(vx, vy)
        if lu < 1e-12 || lv < 1e-12                         # degenerate edge: keep the corner
            push!(pts, (cx, cy))
            continue
        end
        ux, uy = ux / lu, uy / lu
        vx, vy = vx / lv, vy / lv
        α = acos(clamp(ux * vx + uy * vy, -1.0, 1.0)) / 2   # half the corner angle
        if α < 1e-6 || α > π / 2 - 1e-6                     # straight or folded: keep the corner
            push!(pts, (cx, cy))
            continue
        end
        d = min(r / tan(α), lu / 2, lv / 2)                 # never eat more than half an edge
        rr = d * tan(α)                                     # radius actually achievable
        t1 = (cx + d * ux, cy + d * uy)                     # tangency points
        t2 = (cx + d * vx, cy + d * vy)
        bx, by = ux + vx, uy + vy                           # bisector, points inward
        lb = hypot(bx, by)
        ox = cx + (rr / sin(α)) * bx / lb                   # arc centre
        oy = cy + (rr / sin(α)) * by / lb
        a1 = atan(t1[2] - oy, t1[1] - ox)
        a2 = atan(t2[2] - oy, t2[1] - ox)
        Δ = a2 - a1
        if Δ > π                                            # sweep the short way
            Δ -= 2π
        end
        if Δ < -π
            Δ += 2π
        end
        for m in 0:arcpoints
            a = a1 + Δ * m / arcpoints
            push!(pts, (ox + rr * cos(a), oy + rr * sin(a)))
        end
    end
    return pts
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

# First intersection between the ray leaving (cx, cy) at angle θ and a convex polygon.
function _polygon_border(cx, cy, θ, verts)
    dx, dy = cos(θ), sin(θ)
    best = Inf
    n = length(verts)
    for k in 1:n
        (ax, ay) = verts[k]
        (bx, by) = verts[mod1(k + 1, n)]
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-12                                  # ray parallel to this edge
            continue
        end
        t = ((ax - cx) * ey - (ay - cy) * ex) / den          # distance along the ray
        s = ((ax - cx) * dy - (ay - cy) * dx) / den          # position along the edge
        if t >= 0 && -1e-12 <= s <= 1 + 1e-12 && t < best
            best = t
        end
    end
    if !isfinite(best)
        return (cx, cy)
    end
    return (cx + best * dx, cy + best * dy)
end

_hexagon_border(cx, cy, θ, hw, hh) = _polygon_border(cx, cy, θ, _hexagon_vertices(cx, cy, hw, hh))

function _border_point(node::AbstractNode, cx, cy, θ, hw, hh)
    shape = _node_shape(node)
    if shape == :circle
        return _circle_border(cx, cy, θ, hw)
    elseif shape == :rectangle
        return _rect_border(cx, cy, θ, hw, hh)
    else
        # :hexagon and :rounded_hexagon share the sharp border: rounding is cosmetic
        # and never moves the border by more than the corner radius.
        return _hexagon_border(cx, cy, θ, hw, hh)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Node colour, stroke and shape contexts
# ─────────────────────────────────────────────────────────────────────────────

function _node_color(node::AbstractNode)
    if node isa FunctionalNode
        return "orange"
    elseif isprecise(node)
        return "lightgreen"
    else
        return "limegreen"
    end
end

function _node_strokewidth(node::AbstractNode)
    if node isa AbstractContinuousNode && !isempty(node.discretization)
        return 1.2mm   # thick border: discretized continuous node
    else
        return 0.3mm   # standard border
    end
end

function _node_form(node::AbstractNode, x, y, hw, hh)
    shape = _node_shape(node)
    if shape == :circle
        return circle(x, y, hw)
    elseif shape == :rectangle
        return rectangle(x - hw, y - hh, 2hw, 2hh)
    elseif shape == :hexagon
        return polygon(_hexagon_vertices(x, y, hw, hh))
    else
        return polygon(_rounded_polygon(_hexagon_vertices(x, y, hw, hh), _HEX_CORNER_RADIUS * hw))
    end
end

function _build_node_contexts(locs_x, locs_y, node_list, hw, hh)
    ctxs = Compose.Context[]
    for (i, node) in enumerate(node_list)
        push!(ctxs,
            compose(
                context(),
                _node_form(node, locs_x[i], locs_y[i], hw, hh),
                fill(_node_color(node)),
                Compose.stroke("black"),
                linewidth(_node_stroskewidth(node))
            )
        )
    end
    return ctxs
end