include("TB1.jl")
using TB1
using PyPlot
pygui(true)

# n, nratios, Cheb_Moments = Read_DoS("build/toy_model_1d_DoS_3.jld")

# Energies, DofS = DoS(2.55, 0, Cheb_Moments);

# # Conductivities as a function of Fermi Levels
# Image(Energies, DofS, 2*n, 1, (.95, 1.05), (-2.5, -1.), :absolute, (0., .2),
#                 "Ratio of lattice constants", "Energy", "DoS", 1)
# savefig("DoSScan_low_zoom.pdf")

# Image(Energies, DofS, 2*n, 1, (.95, 1.05), (1., 2.5), :absolute, (0., .2),
#                 "Ratio of lattice constants", "Energy", "DoS", 1)
# savefig("DoSScan_high_zoom.pdf")

n, nratios, Cheb_Moments = Read_DoS("build/toy_model_1d_DoS_4.jld")

Energies, DofS = DoS(2.55, 0, Cheb_Moments);

# Conductivities as a function of Fermi Levels
Image(Energies, DofS, 2*n, 1, (.2, .3), (-2.5, 2.5), :relative, (0., .25),
                "Ratio of lattice constants", "Energy", "DoS", 1)
savefig("DoSScan.25.pdf")
