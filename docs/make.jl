using Documenter
using DocumenterCitations
using DocumenterVitepress
using EnhancedBayesianNetworks
using Cairo
using Fontconfig
using DocumenterInterLinks

DocMeta.setdocmeta!(
    EnhancedBayesianNetworks,
    :DocTestSetup,
    :(using EnhancedBayesianNetworks);
    recursive=true,
)

bib=CitationBibliography(joinpath(@__DIR__, "citations.bib"))

links = InterLinks("UncertaintyQuantification" => "https://juliauq.github.io/UncertaintyQuantification.jl/stable/objects.inv")

format = DocumenterVitepress.MarkdownVitepress(
    repo="https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl",
    devbranch="main",
    devurl="dev"
)

pages = [
    "Home" => "index.md",
    "Manual" => [
        "Introduction" => "manual/introduction.md",
        "Getting Started" => "manual/gettingstarted.md",
        "Nodes" => "manual/nodes.md",
        "Networks" => "manual/networks.md",
        "Reduction & Structural Reliability Problem" => "manual/reduction.md",
        "Inference" => "manual/inference.md",
        "Parameter Learning" => "manual/parameterlearning.md",
        "Plotting" => "manual/plotting.md",
    ],
    # "Examples" => [
    #     "Asia Network"     => "examples/asia_bn.md",
    #     "Fire Protection"  => "examples/fire_protection.md",
    #     "Straub Example"   => "examples/straub_example.md",
    # ],
    "API" => [
        "Nodes" => "api/nodes.md",
        "Networks" => "api/networks.md",
        "Inference" => "api/inference.md",
        "Parameter Learning" => "api/learning.md",
        "Plotting" => "api/plotting.md",
    ],
    "References" => "references.md",
]

makedocs(;
    modules=[EnhancedBayesianNetworks],
    plugins=[bib, links],
    authors="Andrea Perin & Jasper Behrensdorf",
    sitename="EnhancedBayesianNetworks.jl",
    pages=pages,
    source="src",
    build="build",
    warnonly=false,
    format=format,
    draft=false,
)

DocumenterVitepress.deploydocs(;
    repo="github.com/JuliaUQ/EnhancedBayesianNetworks.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)