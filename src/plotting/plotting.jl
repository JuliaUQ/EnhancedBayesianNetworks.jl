# Internal base values.
const _BASE_TITLESIZE = 18pt   # title font size
const _BORDER_PAD = 0.12       # fraction of canvas kept free at each edge
const _CM_PER_PT = 2.54 / 72   # one point, in centimetres

include("layout.jl")
include("shapes.jl")
include("edges.jl")
include("labels.jl")
include("legend.jl")

"""
    gplot(net; node_scale, label_size, title, title_scale, figsize, legend, legend_fontsize, legend_x, legend_y)

Draw a network as a layered top-down graph. Nodes are placed by depth: roots on the
first row, every other node one row below its deepest parent. Shape encodes the node
type — circle for continuous, rectangle for discrete, rounded hexagon for continuous
functional, pointy hexagon for discrete functional — while colour encodes precise
(pale) versus imprecise (bright), with functional nodes in orange. A thick border marks
a continuous node carrying a discretization. Discrete nodes also show their number of
states below the name. Edges attach to the exact border of each shape.

`figsize` is the canvas size in centimetres, a `(width, height)` tuple of numbers
(default `(20, 20)`). `label_size` is the node-label font size in points. Pass
`legend=true` to draw the shape/colour key: its top-left corner sits at `(legend_x,
legend_y)` centimetres from the top-left of the canvas, and `legend_fontsize` (in
points) sets its text size — the icons and spacing scale with it. Returns a
`Compose.Context`, which [`saveplot`](@ref) writes to SVG.

# Examples
```julia
W = DiscreteNode(:W)
W[:W => :sunny] = 0.7
W[:W => :rainy] = 0.3
U = ContinuousNode(:U, [:W])
U[:W => :sunny] = Normal()
U[:W => :rainy] = Normal(2, 1)

net = EnhancedBayesianNetwork([W, U])
add_child!(net, W, U)
order!(net)

p = gplot(
    net;
    node_scale = 1.0,           # scale every node shape
    label_size = 8,             # node-label font size, in points
    title = "",            # title text above the graph
    title_scale = 1.0,           # title font scale
    figsize = (20, 20),      # canvas (width, height), in cm
    legend = false,         # draw the shape/colour key
    legend_fontsize = 9,             # legend text size in points; icons scale with it
    legend_x = 13.0,          # legend top-left corner x, in cm from the left
    legend_y = 12.0,          # legend top-left corner y, in cm from the top
    background_color = "transparent", # canvas background colour
)

saveplot(p, "weather.svg")
```
"""
function gplot(
        net::Union{AbstractNetwork, DirectAcyclicGraph};
        node_scale::Float64 = 1.0,
        label_size::Real = 8,
        title::String = "",
        title_scale::Float64 = 1.0,
        figsize::Tuple{Real, Real} = (20, 20),
        legend::Bool = false,
        legend_fontsize::Real = 9,
        legend_x::Real = 13.0,
        legend_y::Real = 12.0,
        background_color::String = "transparent"
    )
    node_list = net.nodes
    n = length(node_list)

    hw = node_scale * 0.05
    hh = hw * 0.95
    al = 0.03 * node_scale

    ew = 0.3mm * node_scale
    ts = _BASE_TITLESIZE * title_scale

    # One point, expressed in canvas units: 1 unit == figsize[2] cm (see below),
    # so anything measured in points converts to units by this factor.
    pt_to_units = _CM_PER_PT / figsize[2]

    # ── positions ────────────────────────────────────────────────────────────
    # Work in an isotropic coordinate system so shapes stay round and arrows stay
    # seated at any figsize: the canvas aspect ratio `ar = width/height` is folded
    # into a UnitBox spanning [0, ar] × [0, 1] (set below), and every x coordinate
    # is spread across that range. One x-unit and one y-unit then map to the same
    # physical length, so all the geometry (radii, angles, border points, arrows)
    # computed downstream is undistorted.
    ar = figsize[1] / figsize[2]
    top_pad = isempty(title) ? 0.12 : 0.18
    locs_x, locs_y = _layered_positions(net.A, _BORDER_PAD, top_pad)
    locs_x = locs_x .* ar

    # ── edges ────────────────────────────────────────────────────────────────
    edge_list = [(i, j) for i in 1:n for j in 1:n if net.A[i, j] != 0]
    edge_lines, edge_arrows = _build_edges(
        edge_list,
        locs_x,
        locs_y,
        node_list,
        hw,
        hh,
        al,
        π / 9
    )

    # ── node shapes and labels ───────────────────────────────────────────────
    node_ctxs = _build_node_contexts(
        locs_x,
        locs_y,
        node_list,
        hw,
        hh
    )
    label_ctxs = _build_labels(
        node_list,
        locs_x,
        locs_y,
        label_size,
        pt_to_units
    )

    # ── optional title ───────────────────────────────────────────────────────
    title_ctx = isempty(title) ? context() : compose(
            context(),
            Compose.text(0.5 * ar, _BORDER_PAD / 2, title, hcenter, vcenter),
            fill("black"),
            fontsize(ts),
            Compose.font("Helvetica")
        )

    # ── assemble (painter's order: back → front) ─────────────────────────────
    Compose.set_default_graphic_size(figsize[1] * cm, figsize[2] * cm)

    legend_ctx = legend ? _build_legend(
            legend_x / figsize[2],   # cm → canvas units (1 unit == figsize[2] cm)
            legend_y / figsize[2],
            legend_fontsize,
            pt_to_units
        ) : context()

    return compose(
        context(units = UnitBox(0, 0, ar, 1)),
        title_ctx,
        label_ctxs...,                                                                  # labels (front)
        node_ctxs...,                                                                   # node shapes
        legend_ctx,
        compose(context(), polygon(edge_arrows), fill("black")),                        # arrowheads
        compose(context(), line(edge_lines), Compose.stroke("black"), linewidth(ew)),   # edge lines (back)
        compose(context(), rectangle(), fill(background_color))                         # background
    )
end

"""
    saveplot(p, filename::String)

Save a gplot result to an SVG file.
"""
function saveplot(p, filename::String)
    return draw(SVG(filename), p)
end
