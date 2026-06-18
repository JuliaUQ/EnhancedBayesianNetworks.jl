using EnhancedBayesianNetworks

weather = DiscreteNode(:W)
weather[:W=>:Cloudy] = 0.5
weather[:W=>:Sunny] = 0.5

rain = DiscreteNode(:R, [:W])
rain[:W=>:Cloudy, :R=>:Yes] = 0.8
rain[:W=>:Cloudy, :R=>:No] = 0.2
rain[:W=>:Sunny, :R=>:Yes] = 0.1
rain[:W=>:Sunny, :R=>:No] = 0.9

sprinkler = DiscreteNode(:S, [:W])
sprinkler[:W=>:Cloudy, :S=>:On] = 0.4
sprinkler[:W=>:Cloudy, :S=>:Off] = 0.4
sprinkler[:W=>:Cloudy, :S=>:broken] = 0.2
sprinkler[:W=>:Sunny, :S=>:On] = 0.6
sprinkler[:W=>:Sunny, :S=>:Off] = 0.3
sprinkler[:W=>:Sunny, :S=>:broken] = 0.1

grass = DiscreteNode(:G, [:S, :R])
grass[:R=>:No, :S=>:On, :G=>:Dry] = 0.2
grass[:R=>:No, :S=>:On, :G=>:Wet] = 0.8
grass[:R=>:No, :S=>:Off, :G=>:Wet] = 0.2
grass[:R=>:No, :S=>:Off, :G=>:Dry] = 0.8
grass[:R=>:No, :S=>:broken, :G=>:Wet] = 0.1
grass[:R=>:No, :S=>:broken, :G=>:Dry] = 0.9
grass[:R=>:Yes, :S=>:On, :G=>:Wet] = 0.6
grass[:R=>:Yes, :S=>:On, :G=>:Dry] = 0.4
grass[:R=>:Yes, :S=>:Off, :G=>:Wet] = 0.55
grass[:R=>:Yes, :S=>:Off, :G=>:Dry] = 0.45
grass[:R=>:Yes, :S=>:broken, :G=>:Wet] = 0.58
grass[:R=>:Yes, :S=>:broken, :G=>:Dry] = 0.42


nodes = [weather, rain, sprinkler, grass]
bn = BayesianNetwork(nodes)
add_child!(bn, :W, :R)
add_child!(bn, :W, :S)
add_child!(bn, :R, :G)
add_child!(bn, :S, :G)
order!(bn)

query = [:W]

evidence = Evidence(:G => :Wet, :R => :Yes)


factors = factorize(bn)

for f in factors
    println(f.vars)
    println(size(f.table))
end


fg = factors[4]
EnhancedBayesianNetworks.sumout(fg, 2)


# test restrict - scalar
f = Factor(
    [1],
    [0.5, 0.5]
)
r = EnhancedBayesianNetworks.restrict(f, 1, 1)
r.vars == Int[]
r.table[] == 0.5

# test restrict - not scalar
f = factors[2]   # W,R
r = EnhancedBayesianNetworks.restrict(f, 2, 1)
r.vars == [1]
size(r.table) == (2,)
r.table[1] == 0.8
r.table[2] == 0.1


# test sumout
f = Factor([1], [0.5, 0.5])
s = EnhancedBayesianNetworks.sumout(f, 1)
s.vars == Int[]
s.table[] == 1.0

s = EnhancedBayesianNetworks.sumout(factors[2], 2)
s.vars == [1]
s.table[1] == 1.0
s.table[2] == 1.0


# expand test - no expansion
f = Factor(
    [1, 2],
    reshape(collect(1:4), 2, 2)
)
allpos = Dict(v => i for (i, v) in enumerate([1, 2]))
A = EnhancedBayesianNetworks.expand(f, [1, 2], allpos)
size(A) == (2, 2)
collect(A) == f.table

# expand test - Add missing variable
f = Factor(
    [2, 4],
    reshape(collect(1:4), 2, 2)
)
allpos = Dict(v => i for (i, v) in enumerate([1, 2, 4]))
A = EnhancedBayesianNetworks.expand(f, [1, 2, 4], allpos)
size(A) == (1, 2, 2)
A[1, 1, 1] == f.table[1, 1]
A[1, 2, 1] == f.table[2, 1]
A[1, 1, 2] == f.table[1, 2]
A[1, 2, 2] == f.table[2, 2]

# expand test - variable order
f = Factor(
    [4, 2],
    reshape(collect(1:4), 2, 2)
)
allpos = Dict(v => i for (i, v) in enumerate([1, 2, 4]))
A = EnhancedBayesianNetworks.expand(f, [1, 2, 4], allpos)

expected = Array{Int}(undef, 1, 2, 2)
expected[1, 1, 1] = 1
expected[1, 1, 2] = 2
expected[1, 2, 1] = 3
expected[1, 2, 2] = 4
all(collect(A) .== expected)


# test multiply


fW = factors[1]
fR = factors[2]

fWR = EnhancedBayesianNetworks.multiply(fW, fR)

fWR.vars == [1, 2]
size(fWR.table) == (2, 2)
fWR.table[1, 1] == 0.4
fWR.table[1, 2] == 0.1
fWR.table[2, 1] == 0.05
fWR.table[2, 2] == 0.45


fS = factors[3]    # vars=[1,3]
fR = factors[2]    # vars=[1,2]

f = EnhancedBayesianNetworks.multiply(fS, fR)

f.vars == [1, 3, 2]
size(f.table) == (2, 3, 2)

# commutativity
A = EnhancedBayesianNetworks.multiply(fW, fR)
B = EnhancedBayesianNetworks.multiply(fR, fW)
sort(A.vars) == sort(B.vars)

# constant factor
c = Factor(Int[], fill(0.5))
f = Factor([1], [0.5, 0.5])
m = EnhancedBayesianNetworks.multiply(c, f)
m.vars == [1]
m.table == [0.25, 0.25]

# multiply all 2
f = EnhancedBayesianNetworks.multiply([fW, fR])
f.vars == [1, 2]
size(f.table) == (2, 2)
f.table[1, 1] == 0.4
f.table[1, 2] == 0.1
f.table[2, 1] == 0.05
f.table[2, 2] == 0.45
f1 = factors[1]
f2 = factors[2]
a = EnhancedBayesianNetworks.multiply([f1, f2])
b = EnhancedBayesianNetworks.multiply(f1, f2)
a.table == b.table
a.vars == b.vars

# multiply all 3
f = EnhancedBayesianNetworks.multiply([
    factors[1],  # W
    factors[2],  # W,R
    factors[3]   # W,S
])
f.vars == [1, 2, 3]
size(f.table) == (2, 2, 3)
f.table[1, 1, 1] ≈ 0.16
f.table[2, 2, 3] ≈ 0.045

# multiply all 1
m = EnhancedBayesianNetworks.multiply([fW])
m.vars == fW.vars
m.table == fW.table

# error message
EnhancedBayesianNetworks.multiply(Factor[])

# normalize 1D
f = Factor(
    [1],
    [2.0, 3.0]
)
n = EnhancedBayesianNetworks.normalize(f)
n.vars == [1]
n.table ≈ [0.4, 0.6]
sum(n.table) ≈ 1.0

# normalize multiD
f = Factor(
    [1, 2],
    [1.0 2.0;
        3.0 4.0]
)
n = EnhancedBayesianNetworks.normalize(f)
sum(n.table) ≈ 1.0
n.table ≈
[
    0.1 0.2;
    0.3 0.4
]

# normalize constant factor
f = Factor(
    Int[],
    fill(5.0)
)
n = EnhancedBayesianNetworks.normalize(f)
n.vars == Int[]
n.table[] ≈ 1.0

# eliminate var
factors = factorize(bn)
newfactors =
    EnhancedBayesianNetworks.eliminate_var(
        factors,
        2
    )
length(newfactors) == 3
all(
    !EnhancedBayesianNetworks.containsvar(f, 2)
    for f in newfactors
)

sort.(getproperty.(newfactors, :vars)) == [[1], [1, 3], [1, 3, 4]]

# empty index
newfactors =
    EnhancedBayesianNetworks.eliminate_var(
        factors,
        99
    )
newfactors === factors

