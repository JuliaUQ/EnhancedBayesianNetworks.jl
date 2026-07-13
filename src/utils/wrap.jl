# Normalise a value to a vector: a scalar becomes a 1-element vector, an array passes through unchanged. 
# Lets APIs accept either a single item or a collection uniformly.
wrap(x) = [x]
wrap(x::AbstractArray) = x