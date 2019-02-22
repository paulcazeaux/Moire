include("TB1.jl")
using TB1
using PyPlot
pygui(true)

n, nratios, Cheb_Moments = Read_DoS("bi_graphene_0.jld")

Cheb_Moments = Cheb_Moments[1:n, :];

Energies, DofS = DoS(11.5, 2, Cheb_Moments)

clf()
plot(Energies, DofS)
# ax = axes()
# ax[:set_xlim]([-1.2, 1.2])
# ax[:set_ylim]([0, 2])

savefig("DoS.bi_graphene_0.5.pdf")