include("TB1.jl")
using TB1
using PyPlot
pygui(true)

n, n_nodes, Cheb_Moments = Read_LDoS("/Users/cazeaux/Dropbox/Workplace/Projets_actuels/Minnesota/Calgebras_1D/Cpp/Output/test/bi_graphene002.jld")

clf()
Energies, DofS = DoS(11.5, 2, Cheb_Moments[29, :])
plot(Energies, DofS)
Energies, DofS = DoS(11.5, 2, Cheb_Moments[30, :])
plot(Energies, DofS)
