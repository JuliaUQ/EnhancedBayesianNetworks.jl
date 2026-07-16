# Normalise a value to a vector: a scalar becomes a 1-element vector, an array passes through unchanged. 
# Lets APIs accept either a single item or a collection uniformly.
_wrap(x) = [x]
_wrap(x::AbstractArray) = x