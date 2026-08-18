function _build_labels(node_list, locs_x, locs_y, label_size, pt_to_units)
    labels = Compose.Context[]
    ls = label_size * pt
    count_ls = 0.8 * label_size * pt
    offset = 0.6 * label_size * pt_to_units   # half a line, in canvas units
    for (i, node) in enumerate(node_list)
        x = locs_x[i]
        y = locs_y[i]
        if node isa AbstractDiscreteNode
            push!(
                labels,
                compose(context(), text(x, y - offset, string(node.name), hcenter, vcenter), fontsize(ls))
            )
            push!(
                labels,
                compose(
                    context(),
                    text(
                        x,
                        y + offset,
                        "[" * string(length(states(node))) * "]",
                        hcenter,
                        vcenter
                    ),
                    fontsize(count_ls)
                )
            )
        else
            push!(
                labels,
                compose(context(), text(x, y, string(node.name), hcenter, vcenter), fontsize(ls))
            )
        end
    end
    return labels
end
