# # Fire Protection Credal Network
#
# This example models fire safety in a building. A fire may produce smoke and set
# off an alarm; the alarm can also be triggered by tampering. Smoke prompts the
# occupants to leave, and neighbours may report the accident. Every conditional
# probability is known only up to an **interval** rather than a single number, so
# the model is a [`CredalNetwork`](@ref) (CN) and inference returns *probability bounds*
# instead of point values.
#
# The network and the probability intervals below are the benchmark of
# Estrada-Lugo, Tolo, De Angelis & Patelli [estrada-lugo_pseudo_2019](@cite)
# (their "small-interval" parameterisation), which lets us check the bounds
# computed here against their published results.

using EnhancedBayesianNetworks

# ## Root nodes
#
# Both roots are imprecise: a recent tampering event, and an actual fire.

T = DiscreteNode(:Tampering)
T[:Tampering => :YesT] = Interval(0.00889, 0.01001)
T[:Tampering => :NoT] = Interval(0.98999, 0.99111)

F = DiscreteNode(:Fire)
F[:Fire => :YesF] = Interval(0.040011, 0.041022)
F[:Fire => :NoF] = Interval(0.958978, 0.959989)

# ## Alarm
#
# The alarm depends on both `Tampering` and `Fire`.

A = DiscreteNode(:Alarm, [:Tampering, :Fire])
A[:Tampering => :YesT, :Fire => :YesF, :Alarm => :YesA] = Interval(0.564106, 0.6)
A[:Tampering => :YesT, :Fire => :YesF, :Alarm => :NoA] = Interval(0.4, 0.435894)
A[:Tampering => :YesT, :Fire => :NoF, :Alarm => :YesA] = Interval(0.880001, 0.9)
A[:Tampering => :YesT, :Fire => :NoF, :Alarm => :NoA] = Interval(0.1, 0.119999)
A[:Tampering => :NoT, :Fire => :YesF, :Alarm => :YesA] = Interval(0.987342, 0.99)
A[:Tampering => :NoT, :Fire => :YesF, :Alarm => :NoA] = Interval(0.01, 0.012658)
A[:Tampering => :NoT, :Fire => :NoF, :Alarm => :YesA] = Interval(0.000003, 0.0002)
A[:Tampering => :NoT, :Fire => :NoF, :Alarm => :NoA] = Interval(0.9998, 0.999997)

# ## Smoke, Leaving and Report
#
# The remaining nodes form a chain: `Fire` → `Smoke`, and `Alarm` → `Leaving` →
# `Report`.

S = DiscreteNode(:Smoke, [:Fire])
S[:Fire => :YesF, :Smoke => :YesS] = Interval(0.89, 0.91)
S[:Fire => :YesF, :Smoke => :NoS] = Interval(0.09, 0.11)
S[:Fire => :NoF, :Smoke => :YesS] = Interval(0.01, 0.102469)
S[:Fire => :NoF, :Smoke => :NoS] = Interval(0.897531, 0.915557)

L = DiscreteNode(:Leaving, [:Alarm])
L[:Alarm => :YesA, :Leaving => :YesL] = Interval(0.870001, 0.9)
L[:Alarm => :YesA, :Leaving => :NoL] = Interval(0.1, 0.129999)
L[:Alarm => :NoA, :Leaving => :YesL] = Interval(0.400001, 0.414423)
L[:Alarm => :NoA, :Leaving => :NoL] = Interval(0.585577, 0.599999)

R = DiscreteNode(:Report, [:Leaving])
R[:Leaving => :YesL, :Report => :YesR] = Interval(0.75, 0.759989)
R[:Leaving => :YesL, :Report => :NoR] = Interval(0.240011, 0.25)
R[:Leaving => :NoL, :Report => :YesR] = Interval(0.171101, 0.190012)
R[:Leaving => :NoL, :Report => :NoR] = Interval(0.809988, 0.828899)

# ## Assembling the Credal Network
#
# Because the nodes are imprecise, they are wrapped in a [`CredalNetwork`](@ref)
# rather than a `BayesianNetwork`.

cn = CredalNetwork([T, F, A, S, L, R])
add_child!(cn, :Tampering, :Alarm)
add_child!(cn, :Fire, :Alarm)
add_child!(cn, :Fire, :Smoke)
add_child!(cn, :Alarm, :Leaving)
add_child!(cn, :Leaving, :Report)
order!(cn)

gplot(cn, background_color = "white", legend = true, label_size = 10, legend_x = 14.5, legend_y = 13.5)

# ## Inference without evidence
#
# For a Credal Network [`infer`](@ref) returns **bounds**: the lowest and highest
# probability of each state across the whole family of networks.

infer(cn, [:Smoke], Evidence())

#-

infer(cn, [:Report], Evidence())

#-

infer(cn, [:Alarm], Evidence())

#-

infer(cn, [:Leaving], Evidence())

# ## Inference with evidence
#
# Conditioning shifts and tightens the bounds. Given that a fire has broken out,
# how likely are the occupants to leave, and to report?

infer(cn, [:Leaving], Evidence(:Fire => :YesF))

#-

infer(cn, [:Report], Evidence(:Fire => :YesF))

#-

infer(cn, [:Report], Evidence(:Alarm => :YesA))

# Reasoning diagnostically: having observed the occupants leaving, what are the
# bounds on there being a fire?

infer(cn, [:Fire], Evidence(:Leaving => :YesL))

# ## Validation
#
# These are exactly the eight queries reported by Estrada-Lugo et al.
# [estrada-lugo_pseudo_2019](@cite) (four without evidence, four with), whose
# "Exact" bounds the values above reproduce.

# ## Timing
#
# As a rough indication, the wall-clock time to answer one query after
# compilation — measured on the machine building these docs, so treat it as
# indicative rather than a controlled benchmark:

infer(cn, [:Fire], Evidence(:Leaving => :YesL))   # warm up (trigger compilation)
elapsed = @elapsed infer(cn, [:Fire], Evidence(:Leaving => :YesL))
println("query solved in ", round(elapsed * 1.0e3; digits = 3), " ms")
