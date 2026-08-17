# Plotting

[`gplot`](@ref) draws a network as a layered, top-down graph: roots on the first row, every other node one row below its deepest parent, with edges attached to the exact border of each shape. 
It works on any network, [Bayesian Network](@ref) (BN), [Credal Network](@ref) (CN), or [Enhanced Bayesian Network](@ref) (eBN), and on a [`DirectAcyclicGraph`](@ref), so a structure can be inspected before its parameters are even [learned](parameterlearning.md).

```@example plotting
using EnhancedBayesianNetworks # hide
W = DiscreteNode(:W)
W[:W => :sunny] = 0.7
W[:W => :rainy] = 0.3
U = ContinuousNode(:U, [:W])
U[:W => :sunny] = Normal()
U[:W => :rainy] = Normal(2, 1)

net = EnhancedBayesianNetwork([W, U])
add_child!(net, W, U)
order!(net)

gplot(net; title = "weather", legend = true, background_color="white")
```

## The visual language

The drawing encodes each node's kind, precision, and discretization, so the diagram doubles as a summary of the model:

| Encoding | Meaning |
|:--|:--|
| **Rectangle** | discrete node |
| **Circle** | continuous node |
| **Pointy hexagon** | discrete functional node |
| **Rounded hexagon** | continuous functional node |
| **Pale fill** | precise node |
| **Bright fill** | imprecise node |
| **Orange fill** | functional node |
| **Thick border** | continuous node carrying a discretization |

Discrete nodes also show their number of states below the name. Pass `legend = true` to draw the shape/colour key on the canvas (positioned by `legend_x` and `legend_y`, as fractions of the canvas).

## Customizing the drawing

[`gplot`](@ref) takes keyword arguments to size and annotate the figure:

- `title`, `title_scale`: a title above the graph and its font scale.
- `node_scale`: scale every node shape up or down.
- `label_scale`: scale the node-label font independently of the shapes.
- `figsize`: the canvas size in centimeters, a `(width, height)` tuple of numbers (default `(20, 20)`).
- `background_color`: the canvas background colour (default `"transparent"`; pass e.g. `"white"` for an opaque figure).
- `legend`, `legend_scale`, `legend_x`, `legend_y`: toggle, scale, and position the legend.

```@example plotting
gplot(net; title = "weather", node_scale = 1.2, label_scale = 0.9, figsize = (16, 12), background_color="white")
```

## Saving to a file

[`gplot`](@ref) returns a `Compose.Context`. [`saveplot`](@ref) writes it to an SVG file:

```julia
p = gplot(net; title = "weather", legend = true)
saveplot(p, "weather.svg")
```
