include("TB1.jl")
using TB1
using PyPlot
pygui(true)

n, n_nodes, Cheb_Moments = Read_LDoS("bi_graphene_0.jld")


clf()
for m=0:5
for n=0:5
    # Energies, DofS = DoS(11.5, 2, Cheb_Moments[(m + 6*n)*2+1, :])
    # plot(Energies, DofS)
    # Energies, DofS = DoS(11.5, 2, Cheb_Moments[(m + 6*n)*2+2, :])
    # plot(Energies, DofS)
    Energies, DofS = DoS(11.5, 2, squeeze(sum(Cheb_Moments[(m + 6*n)*2+(1:2),:], 1), 1)/2)
    plot(Energies, DofS)
 end
end
ax = axes()
ax[:set_xlim]([-2, 2]);
ax[:set_ylim]([0, 2]);
