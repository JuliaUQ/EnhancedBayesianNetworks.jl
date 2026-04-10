# given multiple intervals this function return the sum of the lower and upper bounds
function sum_intervals_and_float(args...)
    lb_sum = 0
    ub_sum = 0

    for x in args
        if isa(x, Interval)
            lb_sum += x.lb
            ub_sum += x.ub
        elseif isa(x, Number)
            lb_sum += x
            ub_sum += x
        end
    end
    return (lb_sum, ub_sum)
end