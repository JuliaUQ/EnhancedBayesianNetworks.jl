flat(x) = [x, x]
flat(x::Interval) = [x.lb, x.ub]