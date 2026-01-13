wrap(x) = [x]
wrap(x::AbstractArray) = x
wrap(x::Parameter) = [x]
wrap(x::AbstractNode) = [x]