# ─────────────────────────────────────────────────────────────────────────────
# Legend: a shape/colour key drawn at a fixed point on the canvas.
#
# Everything scales off one number, the font size. `fontsize_pt` is in points and
# `em` is that same height expressed in canvas units (via `pt_to_units`), so the
# icons, bars and row spacing all track the text automatically — change the font
# and the whole legend grows or shrinks with it. `x`/`y` are the top-left corner,
# also in canvas units (the caller converts them from centimetres).
# ─────────────────────────────────────────────────────────────────────────────
function _build_legend(x, y, fontsize_pt, pt_to_units)
    em = fontsize_pt * pt_to_units      # font height, in canvas units
    fs = fontsize_pt * pt
    header_fs = 1.1 * fontsize_pt * pt

    row_h = 1.6 * em                    # vertical distance between rows
    icon_x = x + 1.0 * em               # icon centre column
    label_x = x + 2.2 * em              # label start column
    r = 0.55 * em                       # icon radius / half-size
    bar_w = 1.1 * em
    bar_h = 0.4 * em
    thin = 0.09 * fontsize_pt * pt    # stroke widths are absolute, but scale with the font
    thick = 0.33 * fontsize_pt * pt

    row_y(i) = y + (i - 0.5) * row_h

    header(i, label) = compose(context(),
        Compose.text(x + 0.3 * em, row_y(i), label, hleft, vcenter), fontsize(header_fs))

    lbl(i, label) = compose(context(),
        Compose.text(label_x, row_y(i), label, hleft, vcenter), fontsize(fs))

    shape(i, formfn, label; fillcolor=nothing, linew=thin) = compose(context(),
        compose(context(), formfn(icon_x, row_y(i)), fill(fillcolor), Compose.stroke("black"), linewidth(linew)),
        lbl(i, label))

    colour(i, label, c) = compose(context(),
        compose(context(), rectangle(icon_x - bar_w / 2, row_y(i) - bar_h / 2, bar_w, bar_h), fill(c)),
        lbl(i, label))

    compose(context(),
        header(1, "Shape"),
        shape(2, (cx, cy) -> circle(cx, cy, r), "Continuous"),
        shape(3, (cx, cy) -> rectangle(cx - r, cy - r, 2r, 2r), "Discrete"),
        shape(4, (cx, cy) -> polygon(_rounded_polygon(_hexagon_vertices(cx, cy, r, r), _HEX_CORNER_RADIUS * r)), "Continuous functional"),
        shape(5, (cx, cy) -> polygon(_hexagon_vertices(cx, cy, r, r)), "Discrete functional"),
        shape(6, (cx, cy) -> circle(cx, cy, r), "To be discretized"; linew=thick),
        header(7, "Color"),
        colour(8, "Precise", "lightgreen"),
        colour(9, "Imprecise", "limegreen"),
        colour(10, "Functional", "orange"),
    )
end
