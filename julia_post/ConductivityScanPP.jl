include("TB1.jl");
using TB1;
using PyPlot
pygui(true)

n, nratios, FermiLevels, Occupancies, Conductivities = Read("ConductivityScan.jld")


GradientImage(FermiLevels, Occupancies, 2*n, 2, :custom_ratios, extrema(FermiLevels), :relative, (0., .2),
  "Ratio of lattice constants", "Energy", "Density of States", 1)
savefig("DoSScan.pdf")

# Conductivities as a function of Fermi Levels
Image(FermiLevels, Conductivities, 2*n, 4, :custom_ratios, extrema(FermiLevels), extrema(Conductivities),
  "Ratio of lattice constants", "Fermi Level", "Conductivity", 2)
savefig("ConductivityScan.pdf")

# Conductivities as a function of Occupancies
Image(Occupancies, Conductivities, 2*n, 4, :custom_ratios, (0.,1.), extrema(Conductivities),
  "Ratio of lattice constants", "Integrated Density of States", "Conductivity", 3)
savefig("ConductivityVsOccupancy.pdf")
