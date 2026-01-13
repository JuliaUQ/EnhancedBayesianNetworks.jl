function gplot(net::AbstractNetwork;
    auto_node_size=false,
    NODESIZEFACTOR=0.1,
    ARROWLENGTH=0.1,
    NODELABELSIZE=4.0,
    EDGELINEWIDTH=1.0 / sqrt(length(net.nodes)),
    DISCRETIZATION_THICKNESS=1.4,
    STROKELENGTH=2mm,
    STROKEDISTANCE=2mm,
    title="",
    title_color="black",
    title_size=4.0,
    font_family="Helvetica",
    nodelabelc="black",
    nodelabelsize=1.0,
    nodelabeldist=0.0,
    nodelabelangleoffset=π / 4.0,
    edgestrokec="black",
    edgelinewidth=1.0,
    arrowangleoffset=π / 9,
    background_color=nothing,
    plot_size=(15, 15),
    leftpad=0,
    rightpad=0,
    toppad=0,
    bottompad=0
)
    ## Title
    title_offset = isempty(title) ? 0 : 0.1 * title_size / 4 #Fix title offset
    title = Compose.text(0, -1.2 - title_offset / 2, title, hcenter, vcenter)
    ## Plot dim
    plot_area = (-1.2, -1.2 - title_offset, +2.4, +2.4 + title_offset)
    max_edgelinewidth = EDGELINEWIDTH / maximum(edgelinewidth)
    edgelinewidth *= max_edgelinewidth

    node_list = net.nodes
    pos = _get_position(net.adj_matrix)
    locs_x = map(p -> p[1], pos)
    locs_y = map(p -> p[2], pos)
    min_x, max_x = extrema(locs_x)
    min_y, max_y = extrema(locs_y)
    function scaler(z, a, b)
        if (a - b) == 0.0
            return 0.5
        else
            return 2.0 * ((z - a) / (b - a)) - 1.0
        end
    end
    map!(z -> scaler(z, min_x, max_x), locs_x, locs_x)
    map!(z -> scaler(z, min_y, max_y), locs_y, locs_y)

    nodecircle = fill(0.4 * 2.4, length(locs_x))
    if auto_node_size
        nodesize = map(n -> length(String(n.name)), node_list)
        nodesize = normalize(nodesize) .* 3
    else
        nodesize = fill(NODESIZEFACTOR, length(locs_x))
    end
    nodecircle = map((x, y) -> x * y, nodecircle, nodesize)
    nodes = circle(locs_x, locs_y, nodecircle)

    nodelabel = map(n -> n.name, node_list)
    if !isnothing(nodelabel)
        text_locs_x = deepcopy(locs_x)
        text_locs_y = deepcopy(locs_y)
        texts = Compose.text(text_locs_x .+ nodesize .* (nodelabeldist * cos(nodelabelangleoffset)),
            text_locs_y .- nodesize .* (nodelabeldist * sin(nodelabelangleoffset)),
            map(string, nodelabel), [hcenter], [vcenter])
    end
    max_nodelabelsize = NODELABELSIZE / maximum(nodelabelsize)
    nodelabelsize *= max_nodelabelsize

    edges_list = _get_edges(net.adj_matrix)
    lines, larrows = _build_straight_edges(edges_list, locs_x, locs_y, nodesize, ARROWLENGTH, arrowangleoffset)

    function _is2discretize(n)
        if isa(n, AbstractContinuousNode)
            if !isempty(n.discretization.intervals)
                return true
            else
                return false
            end
        else
            return false
        end
    end

    function _get_stroke(node)
        if _is2discretize(node)
            return (DISCRETIZATION_THICKNESS, "black", [])
        elseif !isa(node, FunctionalNode) && !isprecise(node)
            return (DISCRETIZATION_THICKNESS, "black", [STROKELENGTH, STROKEDISTANCE])
        else
            return (4.0, nothing, [])
        end
    end

    nodestrokelw = map(n -> _get_stroke(n)[1], node_list)
    nodestrokec = map(n -> _get_stroke(n)[2], node_list)
    nodestroked = map(n -> _get_stroke(n)[3], node_list)

    colors = _node_color.(node_list)

    plot_size = (plot_size[1]cm, plot_size[2]cm)
    padding = [leftpad, rightpad, toppad, bottompad]
    leftpad = padding[1]mm
    rightpad = padding[2]mm
    toppad = padding[3]mm
    bottompad = padding[4]mm

    Compose.set_default_graphic_size(plot_size...)
    compose(
        context(units=UnitBox(plot_area...; leftpad, rightpad, toppad, bottompad)),
        compose(context(), title, fill(title_color), fontsize(title_size), Compose.font(font_family)),
        compose(context(), texts, fill(nodelabelc), fontsize(nodelabelsize), Compose.font(font_family)),
        compose(context(), nodes, fill(colors), Compose.stroke(nodestrokec), linewidth(nodestrokelw), strokedash(nodestroked)),
        compose(context(), larrows, fill(edgestrokec)),
        compose(context(), lines, Compose.stroke(edgestrokec), linewidth(edgelinewidth)),
        compose(context(units=UnitBox(plot_area...)), rectangle(plot_area...), fill(background_color))
    )
end

compose(context(), circle(0.5, 0.5, 0.4), fill("red"), Compose.stroke("black"), linewidth(4), strokedash([]))


function _get_position(adj_matrix::SparseMatrixCSC)
    pos = spring(adj_matrix; iterations=1000)
    return pos
end

function _get_edges(adj_matrix::SparseMatrixCSC)
    n = size(adj_matrix)
    edge_list = Vector{Tuple{Int64,Int64}}()
    for i in range(1, n[1])
        for j in range(1, n[2])
            if adj_matrix[i, j] != 0
                push!(edge_list, (i, j))
            end
        end
    end
    return edge_list
end

function _build_straight_edges(edge_list, locs_x, locs_y, nodesize, ARROWLENGTH, arrowangleoffset)
    if ARROWLENGTH > 0.0
        lines_cord, arrows_cord = _graphline(edge_list, locs_x, locs_y, nodesize, ARROWLENGTH, arrowangleoffset)
        lines = line(lines_cord)
        larrows = polygon(arrows_cord)
    else
        lines_cord = _graphline(edge_list, locs_x, locs_y, nodesize)
        lines = line(lines_cord)
        larrows = nothing
    end
    return lines, larrows
end

function _graphline(edge_list, locs_x, locs_y, nodesize::Vector{T}) where {T<:Real}
    num_edges = length(edge_list)
    lines = Array{Vector{Tuple{Float64,Float64}}}(undef, num_edges)
    for (e_idx, e) in enumerate(edge_list)
        i = e[1]
        j = e[2]
        Δx = locs_x[j] - locs_x[i]
        Δy = locs_y[j] - locs_y[i]
        θ = atan(Δy, Δx)
        startx = locs_x[i] + nodesize[i] * cos(θ)
        starty = locs_y[i] + nodesize[i] * sin(θ)
        endx = locs_x[j] + nodesize[j] * cos(θ + π)
        endy = locs_y[j] + nodesize[j] * sin(θ + π)
        lines[e_idx] = [(startx, starty), (endx, endy)]
    end
    lines
end

function _graphline(edge_list, locs_x, locs_y, nodesize::Vector{T}, arrowlength, angleoffset) where {T<:Real}
    num_edges = length(edge_list)
    lines = Array{Vector{Tuple{Float64,Float64}}}(undef, num_edges)
    arrows = Array{Vector{Tuple{Float64,Float64}}}(undef, num_edges)
    for (e_idx, e) in enumerate(edge_list)
        i = e[1]
        j = e[2]
        Δx = locs_x[j] - locs_x[i]
        Δy = locs_y[j] - locs_y[i]
        θ = atan(Δy, Δx)
        startx = locs_x[i] + nodesize[i] * cos(θ)
        starty = locs_y[i] + nodesize[i] * sin(θ)
        endx = locs_x[j] + nodesize[j] * cos(θ + π)
        endy = locs_y[j] + nodesize[j] * sin(θ + π)
        arr1, arr2 = _arrowcoords(θ, endx, endy, arrowlength, angleoffset)
        endx0, endy0 = _midpoint(arr1, arr2)
        e_idx2 = findfirst(==((j, i)), edge_list)
        if !isnothing(e_idx2) && e_idx2 < e_idx
            startx, starty = _midpoint(arrows[e_idx2][[1, 3]]...)
            lines[e_idx2][1] = (endx0, endy0)
        end
        lines[e_idx] = [(startx, starty), (endx0, endy0)]
        arrows[e_idx] = [arr1, (endx, endy), arr2]
    end
    lines, arrows
end

function _node_color(n::AbstractNode)
    if isa(n, FunctionalNode)
        if isa(n, DiscreteFunctionalNode)
            return "lightsalmon"
        else
            return "red1"
        end
    elseif isa(n, ContinuousNode)
        if !isprecise(n)
            return "cyan1"
        else
            return "paleturquoise"
        end
    elseif isa(n, DiscreteNode)
        if !isprecise(n)
            return "green1"
        else
            return "palegreen"
        end
    end
end

function _arrowcoords(θ, endx, endy, arrowlength, angleoffset=20.0 / 180.0 * π)
    arr1x = endx - arrowlength * cos(θ + angleoffset)
    arr1y = endy - arrowlength * sin(θ + angleoffset)
    arr2x = endx - arrowlength * cos(θ - angleoffset)
    arr2y = endy - arrowlength * sin(θ - angleoffset)
    return (arr1x, arr1y), (arr2x, arr2y)
end

function _midpoint(pt1, pt2)
    x = (pt1[1] + pt2[1]) / 2
    y = (pt1[2] + pt2[2]) / 2
    return x, y
end

function saveplot(gplot, filename::String)
    draw(SVG(filename), gplot)
    return nothing
end