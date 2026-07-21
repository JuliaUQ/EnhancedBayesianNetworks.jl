
function _build_labels(node_list, locs_x, locs_y, labelsize, label_scale)
    labels = Compose.Context[]
    for (i, node) in enumerate(node_list)
        x = locs_x[i]
        y = locs_y[i]
        name_offset = 0.008 * label_scale
        count_offset = 0.008 * label_scale
        if node isa AbstractDiscreteNode
            push!(
                labels,
                compose(context(), text(x, y - name_offset, string(node.name), hcenter, vcenter), fontsize(labelsize))
            )
            push!(
                labels,
                compose(
                    context(),
                    text(
                        x,
                        y + count_offset,
                        "["*string(length(states(node)))*"]",
                        hcenter,
                        vcenter
                    ),
                    fontsize(0.8labelsize)
                )
            )
        else
            push!(
                labels,
                compose(context(), text(x, y, string(node.name), hcenter, vcenter), fontsize(labelsize))
            )
        end
    end
    return labels
end