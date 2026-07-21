
# Internal base values — all user parameters scale these.
const _BASE_LABELSIZE = 8pt    # node label font size
const _BASE_TITLESIZE = 18pt   # title font size
const _BORDER_PAD = 0.12       # fraction of canvas kept free at each edge

include("layout.jl")
include("shapes.jl")
include("edges.jl")
include("labels.jl")
include("legend.jl")

"""
    gplot(net; node_scale, label_scale, title, title_scale, figsize, legend, legend_scale, legend_x, legend_y)

Draw a network as a layered top-down graph. Nodes are placed by depth: roots on the
first row, every other node one row below its deepest parent. Shape encodes the node
type — circle for continuous, rectangle for discrete, rounded hexagon for continuous
functional, pointy hexagon for discrete functional — while colour encodes precise
(pale) versus imprecise (bright), with functional nodes in orange. A thick border marks
a continuous node carrying a discretization. Discrete nodes also show their number of
states below the name. Edges attach to the exact border of each shape.

Pass `legend=true` to draw the shape/colour key, positioned by `legend_x` and `legend_y`
as fractions of the canvas. Returns a `Compose.Context`, which [`saveplot`](@ref) writes
to SVG.

# Examples
```julia
W = DiscreteNode(:W)
W[:W=>:sunny] = 0.7
W[:W=>:rainy] = 0.3
U = ContinuousNode(:U, [:W])
U[:W=>:sunny] = Normal()
U[:W=>:rainy] = Normal(2, 1)

net = EnhancedBayesianNetwork([W, U])
add_child!(net, W, U)
order!(net)

p = gplot(net; title="weather", legend=true)
saveplot(p, "weather.svg")
```
"""
function gplot(net::Union{AbstractNetwork,DirectAcyclicGraph};
    node_scale::Float64=1.0,
    label_scale::Float64=1.0,
    title::String="",
    title_scale::Float64=1.0,
    figsize::Tuple=(20cm, 20cm),
    legend::Bool=false,
    legend_scale::Float64=1.0,
    legend_x::Float64=0.72,
    legend_y::Float64=0.62
)
    node_list = net.nodes
    n = length(node_list)

    hw = node_scale * 0.05
    hh = hw * 0.95
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
        ls,
        label_scale
    )

    # ── optional title ───────────────────────────────────────────────────────
    title_ctx = isempty(title) ? context() : compose(
        context(),
        Compose.text(0.5, _BORDER_PAD / 2, title, hcenter, vcenter),
        fill("black"),
        fontsize(ts),
        Compose.font("Helvetica")
    )

    # ── assemble (painter's order: back → front) ─────────────────────────────
    Compose.set_default_graphic_size(figsize[1], figsize[2])

    legend_ctx = legend ? _build_legend(
        legend_scale;
        x_fraction=legend_x,
        y_fraction=legend_y
    ) : context()

    compose(context(),
        title_ctx,
        label_ctxs...,                                                                  # labels (front)
        node_ctxs...,                                                                   # node shapes
        legend_ctx,
        compose(context(), polygon(edge_arrows), fill("black")),                        # arrowheads
        compose(context(), line(edge_lines), Compose.stroke("black"), linewidth(ew)),   # edge lines (back)
    )
end

"""
    saveplot(p, filename::String)

Save a gplot result to an SVG file.
"""
function saveplot(p, filename::String)
    draw(SVG(filename), p)
end