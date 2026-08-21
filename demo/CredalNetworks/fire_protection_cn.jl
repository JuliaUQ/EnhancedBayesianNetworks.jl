using EnhancedBayesianNetworks

T = DiscreteNode(:Tampering)
T[:Tampering=>:YesT] = Interval(0.00889, 0.01001)
T[:Tampering=>:NoT] = Interval(0.98999, 0.99111)

F = DiscreteNode(:Fire)
F[:Fire=>:YesF] = Interval(0.040011, 0.041022)
F[:Fire=>:NoF] = Interval(0.958978, 0.959989)

A = DiscreteNode(:Alarm, [:Tampering, :Fire])
A[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.564106, 0.600000)
A[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.400000, 0.435894)
A[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.880001, 0.900000)
A[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.100000, 0.119999)
A[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.987342, 0.990000)
A[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.010000, 0.012658)
A[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.000003, 0.000200)
A[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.999800, 0.999997)

S = DiscreteNode(:Smoke, [:Fire])
S[:Fire=>:YesF, :Smoke=>:YesS] = Interval(0.890000, 0.910000)
S[:Fire=>:YesF, :Smoke=>:NoS] = Interval(0.090000, 0.110000)
S[:Fire=>:NoF, :Smoke=>:YesS] = Interval(0.010000, 0.102469)
S[:Fire=>:NoF, :Smoke=>:NoS] = Interval(0.897531, 0.915557)

L = DiscreteNode(:Leaving, [:Alarm])
L[:Alarm=>:YesA, :Leaving=>:YesL] = Interval(0.870001, 0.900000)
L[:Alarm=>:YesA, :Leaving=>:NoL] = Interval(0.100000, 0.129999)
L[:Alarm=>:NoA, :Leaving=>:YesL] = Interval(0.400001, 0.414423)
L[:Alarm=>:NoA, :Leaving=>:NoL] = Interval(0.585577, 0.599999)

R = DiscreteNode(:Report, [:Leaving])
R[:Leaving=>:YesL, :Report=>:YesR] = Interval(0.750000, 0.759989)
R[:Leaving=>:YesL, :Report=>:NoR] = Interval(0.240011, 0.250000)
R[:Leaving=>:NoL, :Report=>:YesR] = Interval(0.171101, 0.190012)
R[:Leaving=>:NoL, :Report=>:NoR] = Interval(0.809988, 0.828899)

cn = CredalNetwork([T, F, A, S, L, R])
add_child!(cn, :Tampering, :Alarm)
add_child!(cn, :Fire, :Alarm)
add_child!(cn, :Fire, :Smoke)
add_child!(cn, :Alarm, :Leaving)
add_child!(cn, :Leaving, :Report)
order!(cn)

gplot(cn, background_color="white", legend=true, label_size=10, legend_x=14.5, legend_y=13.5)

infer(cn, [:Smoke], Evidence())

infer(cn, [:Report], Evidence())

infer(cn, [:Alarm], Evidence())

infer(cn, [:Leaving], Evidence())

infer(cn, [:Leaving], Evidence(:Fire => :YesF))

infer(cn, [:Report], Evidence(:Fire => :YesF))

infer(cn, [:Report], Evidence(:Alarm => :YesA))

infer(cn, [:Fire], Evidence(:Leaving => :YesL))

infer(cn, [:Fire], Evidence(:Leaving => :YesL))   # warm up (trigger compilation)
elapsed = @elapsed infer(cn, [:Fire], Evidence(:Leaving => :YesL))
println("query solved in ", round(elapsed * 1e3; digits=3), " ms")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
