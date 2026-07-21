using Documenter
using DocumenterCitations
using DocumenterVitepress
using EnhancedBayesianNetworks

bib=CitationBibliography(joinpath(@__DIR__, "citations.bib"))

format = DocumenterVitepress.MarkdownVitepress(
    repo="https://github.com/JuliaUQ/EnhancedBayesianNetworks.jl",
    devbranch="main",
    devurl="dev")

pages=["Home" => "index.md", "References" => "references.md"]

makedocs(; modules=[EnhancedBayesianNetworks], plugins=[bib], authors="Andrea Perin & Jasper Behrensdorf", pages=pages, sitename="EnhancedBayesianNetworks.jl", source="src", build="build", warnonly=true, format=format, draft=false)

DocumenterVitepress.deploydocs(;
    repo="github.com/JuliaUQ/EnhancedBayesianNetworks.jl",
    target=joinpath(@__DIR__, "build"),
    branch="gh-pages",
    devbranch="main",
    push_preview=true,
)