using Literate

const LITERATE_DIR = "./docs/literate"
const EXAMPLES_DIR = "./docs/src/examples"

"""
    convertExample(jl_path::String, out_dir::String; documenter::Bool = true)

Render a single Literate `.jl` example to its own markdown page in `out_dir`
(one page per file, so each example is an individual page you can name and link
explicitly from `make.jl`). For `documenter=true` pages, drop the leading
`@meta` block that Literate prepends.
"""
function convertExample(jl_path::String, out_dir::String; documenter::Bool = true)
    name = first(splitext(basename(jl_path)))
    mkpath(out_dir)

    Literate.markdown(jl_path, out_dir; documenter = documenter, name = name)

    # remove @meta block created by literate
    if documenter
        md_file = joinpath(out_dir, "$name.md")
        lines = readlines(md_file; keep = true)
        open(md_file, "w") do file
            write.(file, lines[5:end])
        end
    end

    return nothing
end

# docs/src/examples is fully generated (and gitignored): start from a clean
# slate so stale pages from removed/renamed examples never linger and trip
# `warnonly=false`.
rm(EXAMPLES_DIR; recursive = true, force = true)
mkpath(EXAMPLES_DIR)

for (root, _, files) in walkdir(LITERATE_DIR)
    jl_files = filter(f -> endswith(f, ".jl"), files)
    isempty(jl_files) && continue

    rel = relpath(root, LITERATE_DIR)   # e.g. "BayesianNetworks"
    rel == "." && continue              # skip the literate root itself

    out_dir = joinpath(EXAMPLES_DIR, rel)
    documenter = !(basename(root) in ("hpc", "external"))
    for jl in jl_files
        convertExample(joinpath(root, jl), out_dir; documenter = documenter)
    end
end
