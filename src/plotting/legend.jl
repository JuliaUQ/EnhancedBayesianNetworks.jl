
const _LEGEND_X_HEAD = 0.06    # x of a section header, in legend-box fractions
const _LEGEND_X_ICON = 0.11    # x of a row's icon
const _LEGEND_X_LABEL = 0.21   # x where a row's text starts

# Vertical placement of legend row `i` (1-based, headers included).
_legend_y(i::Int) = 0.05 + (i - 1) * 0.098

function _legend_label(y, label, fs)
    compose(context(), Compose.text(_LEGEND_X_LABEL, y, label, hleft, vcenter), fontsize(fs * pt))
end

_legend_header(i::Int, label, fs) = compose(context(),
    Compose.text(_LEGEND_X_HEAD, _legend_y(i), label, hleft, vcenter), fontsize(fs * pt))

# A shape row: `formfn` receives the row's y and returns the outlined icon.
function _legend_shape(formfn, i::Int, label, fs; fillcolor=nothing, linew=0.3mm)
    y = _legend_y(i)
    compose(context(),
        compose(context(), formfn(y), fill(fillcolor), Compose.stroke("black"), linewidth(linew)),
        _legend_label(y, label, fs))
end

# A colour row: a filled bar instead of a shape icon.
function _legend_color(i::Int, label, fs, colour, bar_w, bar_h)
    y = _legend_y(i)
    compose(context(),
        compose(context(), rectangle(_LEGEND_X_ICON - bar_w / 2, y - bar_h / 2, bar_w, bar_h), fill(colour)),
        _legend_label(y, label, fs))
end

function _build_legend(scale; x_fraction=0.72, y_fraction=0.62)
    r = 0.035 * scale
    hw = 0.035 * scale
    hh = 0.026 * scale
    bar_w = 0.07 * scale
    bar_h = 0.014 * scale
    fs = 9 * scale
    header_fs = 10 * scale
    x = _LEGEND_X_ICON

    compose(context(x_fraction, y_fraction, 0.26, 0.34),
        compose(context(), rectangle(), fill(nothing), Compose.stroke("black"), linewidth(0.3mm)),
        _legend_header(1, "Shape", header_fs),
        _legend_shape(y -> circle(x, y, r), 2, "Continuous", fs),
        _legend_shape(y -> rectangle(x - hw, y - hh, 2hw, 2hh), 3, "Discrete", fs),
        _legend_shape(y -> polygon(_rounded_polygon(_hexagon_vertices(x, y, hw, hh), _HEX_CORNER_RADIUS * hw)),
            4, "Continuous functional", fs),
        _legend_shape(y -> polygon(_hexagon_vertices(x, y, hw, hh)), 5, "Discrete functional", fs),
        _legend_shape(y -> circle(x, y, r), 6, "Discretized", fs; fillcolor="lightgreen", linew=1.2mm * scale),
        _legend_header(7, "Color", header_fs),
        _legend_color(8, "Precise", fs, "lightgreen", bar_w, bar_h),
        _legend_color(9, "Imprecise", fs, "limegreen", bar_w, bar_h),
        _legend_color(10, "Functional", fs, "orange", bar_w, bar_h),
    )
end