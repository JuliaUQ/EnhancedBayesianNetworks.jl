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

# Coordinates of every edge: the segment from the source border to the arrowhead
# base, and the arrowhead triangle whose tip sits on the target border. The line
# stops at the arrowhead base so it does not show through the filled triangle.
function _build_edges(edge_list, locs_x, locs_y, nodes, hw, hh,
    arrowlength, angleoffset)
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

        arr1, arr2 = _arrowcoords(θ, tip_x, tip_y, arrowlength, angleoffset)
        lines[e_idx] = [(startx, starty), _midpoint(arr1, arr2)]
        arrows[e_idx] = [arr1, (tip_x, tip_y), arr2]
    end

    return lines, arrows
end