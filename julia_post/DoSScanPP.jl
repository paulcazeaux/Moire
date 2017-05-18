include("TB1.jl");
using TB1;
using PyPlot
pygui(true)

n, nratios, Cheb_Moments = Read_DoS("build/toy_model_1d_DoS.jld")

Energies, DofS = DoS(2.55, 0, Cheb_Moments);

# Conductivities as a function of Fermi Levels
Image(Energies, DofS, 2*n, 4, (.95, 1.05), (-2.5, 2.5), :auto, (),
                "Ratio of lattice constants", "Energy level", "DoS", 1)
savefig("DoSScan.pdf")