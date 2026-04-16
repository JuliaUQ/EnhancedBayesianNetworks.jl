abstract type AbstractNetwork end

include("../utils/topologically_sort.jl")
include("../utils/require_unique.jl")
include("../utils/sum_intervals_and_float.jl")
include("bn/bayesnet.jl")
include("cn/credalnet.jl")
include("networks_common.jl")
include("ebn/ebn.jl")
include("dispatch.jl")