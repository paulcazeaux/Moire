include("TB1.jl")
using TB1
using PyPlot
# pygui(true)

n, nratios, Cheb_Moments = Read_DoS("input.jld")

Energies, DofS = DoS(15., 0.3504, Cheb_Moments)

# Density of States as a function of Energy
Image(Energies, DofS, 2*n, 1, (.25, .75), (-2.5, 2.5), :relative, (0., .25),
                "Ratio of lattice constants", "Energy", "DoS", 1)
savefig("output.pdf")
