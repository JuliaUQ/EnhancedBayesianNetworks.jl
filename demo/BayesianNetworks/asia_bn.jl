using EnhancedBayesianNetworks

A = DiscreteNode(:A)                 # visit to Asia
A[:A => :A_yes] = 0.01
A[:A => :A_no] = 0.99

S = DiscreteNode(:S)                 # smoker
S[:S => :S_yes] = 0.5
S[:S => :S_no] = 0.5

T = DiscreteNode(:T, [:A])           # tuberculosis
T[:A => :A_yes, :T => :T_yes] = 0.05
T[:A => :A_yes, :T => :T_no] = 0.95
T[:A => :A_no, :T => :T_yes] = 0.01
T[:A => :A_no, :T => :T_no] = 0.99

L = DiscreteNode(:L, [:S])           # lung cancer
L[:S => :S_yes, :L => :L_yes] = 0.1
L[:S => :S_yes, :L => :L_no] = 0.9
L[:S => :S_no, :L => :L_yes] = 0.01
L[:S => :S_no, :L => :L_no] = 0.99

B = DiscreteNode(:B, [:S])           # bronchitis
B[:S => :S_yes, :B => :B_yes] = 0.6
B[:S => :S_yes, :B => :B_no] = 0.4
B[:S => :S_no, :B => :B_yes] = 0.3
B[:S => :S_no, :B => :B_no] = 0.7

E = DiscreteNode(:E, [:T, :L])       # tuberculosis OR lung cancer
E[:T => :T_yes, :L => :L_yes, :E => :E_yes] = 1.0
E[:T => :T_yes, :L => :L_yes, :E => :E_no] = 0.0
E[:T => :T_yes, :L => :L_no, :E => :E_yes] = 1.0
E[:T => :T_yes, :L => :L_no, :E => :E_no] = 0.0
E[:T => :T_no, :L => :L_yes, :E => :E_yes] = 1.0
E[:T => :T_no, :L => :L_yes, :E => :E_no] = 0.0
E[:T => :T_no, :L => :L_no, :E => :E_yes] = 0.0
E[:T => :T_no, :L => :L_no, :E => :E_no] = 1.0

X = DiscreteNode(:X, [:E])           # positive X-ray
X[:E => :E_yes, :X => :X_yes] = 0.98
X[:E => :E_yes, :X => :X_no] = 0.02
X[:E => :E_no, :X => :X_yes] = 0.05
X[:E => :E_no, :X => :X_no] = 0.95

D = DiscreteNode(:D, [:E, :B])       # dyspnoea
D[:E => :E_yes, :B => :B_yes, :D => :D_yes] = 0.9
D[:E => :E_yes, :B => :B_yes, :D => :D_no] = 0.1
D[:E => :E_yes, :B => :B_no, :D => :D_yes] = 0.7
D[:E => :E_yes, :B => :B_no, :D => :D_no] = 0.3
D[:E => :E_no, :B => :B_yes, :D => :D_yes] = 0.8
D[:E => :E_no, :B => :B_yes, :D => :D_no] = 0.2
D[:E => :E_no, :B => :B_no, :D => :D_yes] = 0.1
D[:E => :E_no, :B => :B_no, :D => :D_no] = 0.9

bn = BayesianNetwork([A, S, T, L, B, E, X, D])
add_child!(bn, :A, :T)
add_child!(bn, :S, :L)
add_child!(bn, :S, :B)
add_child!(bn, :T, :E)
add_child!(bn, :L, :E)
add_child!(bn, :E, :X)
add_child!(bn, :E, :D)
add_child!(bn, :B, :D)
order!(bn)

gplot(bn, background_color = "white", legend = true, label_size = 10, node_scale = 0.8, legend_x = 5.5, legend_y = 1)

infer(bn, :D, Evidence())

infer(bn, :D, Evidence(:A => :A_yes, :S => :S_yes))

infer(bn, :L, Evidence(:X => :X_yes))

infer(bn, :D, Evidence(:A => :A_yes, :S => :S_yes))   # warm up (trigger compilation)
elapsed = @elapsed infer(bn, :D, Evidence(:A => :A_yes, :S => :S_yes))
println("query solved in ", round(elapsed * 1.0e3; digits = 3), " ms")

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
