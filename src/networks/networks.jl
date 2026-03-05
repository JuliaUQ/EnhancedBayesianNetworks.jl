abstract type AbstractNetwork end

include("../utils/topologically_sort.jl")
include("../utils/require_unique.jl")
include("../utils/sum_intervals_and_float.jl")
include("networks_common.jl")
include("ebn/ebn.jl")

# include("bn/bayesnet.jl")
# include("bn/bayesnet2be.jl")
# include("cn/credalnet.jl")
# include("dispatch.jl")