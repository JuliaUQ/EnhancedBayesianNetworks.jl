# Return a 2-element [lower, upper] for a value: a precise value collapses to [x, x], an Interval expands to its bounds. 
# Used to assemble the bound vectors of imprecise/credal constraints.
_flat(x) = [x, x]
_flat(x::Interval) = [x.lb, x.ub]