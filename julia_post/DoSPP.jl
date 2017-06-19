include("TB1.jl")
using TB1
using PyPlot
pygui(true)

n, nratios, Cheb_Moments = Read_DoS("bi_graphene001.jld")

Energies, DofS = DoS(11.5, -2, Cheb_Moments)

plot(Energies, DofS)
savefig("DoS.bi_graphene001.pdf")
