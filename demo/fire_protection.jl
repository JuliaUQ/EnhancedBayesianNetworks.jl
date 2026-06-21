using EnhancedBayesianNetworks
using Cairo

T = DiscreteNode(:Tampering)
T[:Tampering=>:YesT] = 0.98
T[:Tampering=>:NoT] = 0.02

F = DiscreteNode(:Fire)
F[:Fire=>:YesF] = Interval(0.98, 0.99)
F[:Fire=>:NoF] = Interval(0.01, 0.02)

A = DiscreteNode(:Alarm, [:Tampering, :Fire])
A[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.4, 0.6)
A[:Tampering=>:YesT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.4, 0.5)
A[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.85, 0.9)
A[:Tampering=>:YesT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.1, 0.15)
A[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:YesA] = Interval(0.985, 0.99)
A[:Tampering=>:NoT, :Fire=>:YesF, :Alarm=>:NoA] = Interval(0.01, 0.015)
A[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:YesA] = Interval(0.0001, 0.0002)
A[:Tampering=>:NoT, :Fire=>:NoF, :Alarm=>:NoA] = Interval(0.9998, 0.9999)

S = DiscreteNode(:Smoke, [:Fire])
S[:Fire=>:YesF, :Smoke=>:YesS] = Interval(0.87, 0.91)
S[:Fire=>:YesF, :Smoke=>:NoS] = Interval(0.09, 0.13)
S[:Fire=>:NoF, :Smoke=>:YesS] = Interval(0.01, 0.1)
S[:Fire=>:NoF, :Smoke=>:NoS] = Interval(0.9, 0.99)

L = DiscreteNode(:Leaving, [:Alarm])
L[:Alarm=>:YesA, :Leaving=>:YesL] = Interval(0.88, 0.99)
L[:Alarm=>:YesA, :Leaving=>:NoL] = Interval(0.001, 0.42)
L[:Alarm=>:NoA, :Leaving=>:YesL] = Interval(0.1, 0.12)
L[:Alarm=>:NoA, :Leaving=>:NoL] = Interval(0.58, 0.99)

R = DiscreteNode(:Report, [:Leaving])
R[:Leaving=>:YesL, :Report=>:YesR] = Interval(0.25, 0.76)
R[:Leaving=>:YesL, :Report=>:NoR] = Interval(0.24, 0.75)
R[:Leaving=>:NoL, :Report=>:YesR] = Interval(0.01, 0.2)
R[:Leaving=>:NoL, :Report=>:NoR] = Interval(0.8, 0.99)

nodes = [T, F, A, S, L, R]

cn = CredalNetwork(nodes)
add_child!(cn, :Tampering, :Alarm)
add_child!(cn, :Fire, :Alarm)
add_child!(cn, :Fire, :Smoke)
add_child!(cn, :Alarm, :Leaving)
add_child!(cn, :Leaving, :Report)
order!(cn)

plt1 = gplot(cn)

evidence = Evidence()
query = [:Smoke]
ϕ1 = infer(cn, query, evidence)

evidence = Evidence()
query = [:Report]
ϕ2 = infer(cn, query, evidence)

evidence = Evidence()
query = [:Alarm]
ϕ3 = infer(cn, query, evidence)

evidence = Evidence(
    :Fire => :YesF
)
query = [:Leaving]
ϕ4 = infer(cn, query, evidence)

evidence = Evidence(
    :Fire => :YesF
)
query = [:Report]
ϕ5 = infer(cn, query, evidence)

evidence = Evidence(
    :Leaving => :YesL
)
query = [:Fire]
ϕ6 = infer(cn, query, evidence)

smoke_pot = ϕ1.potential
reference_smoke = [[0.8838 0.9814], [0.0186, 0.1162]]

report_pot = ϕ2.potential
reference_report = [[0.5547, 0.9719], [0.0281, 0.4453]]

alarm_pot = ϕ3.potential
reference_alarm = [[0.9625 0.9733], [0.0281, 0.0375]]

leaving_pot = ϕ4.potential
reference_leaving = [[0.1085 0.1435], [0.8565, 0.8915]]