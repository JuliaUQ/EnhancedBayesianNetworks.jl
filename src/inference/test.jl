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

evidence = Evidence(:G => :Wet, :R => :yes)


factors = factorize(bn)

for f in factors
    println(f.vars)
    println(size(f.table))
end


fg = factors[4]
EnhancedBayesianNetworks.sumout(fg, 2)

