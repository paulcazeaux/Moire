include("TB1.jl")
using TB1
using PyPlot
pygui(true)

n, n_nodes, Cheb_Moments = Read_LDoS("bi_graphene_0.jld")
E_rescale = 11.5;


fig = figure(1)
clf()
ax = subplot(121)

for m=0:5
for n=0:5
    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[(m + 6*n)*2+1, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[(m + 6*n)*2+2, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    Energies, DofS = DoS(E_rescale, 2, squeeze(sum(Cheb_Moments[(m + 6*n)*2+(1:2),:], 1), 1)/2)
    plot(Energies, DofS, linewidth=0.5)

    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[72+(m + 6*n)*2+1, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[72+(m + 6*n)*2+2, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    Energies, DofS = DoS(E_rescale, 2, squeeze(sum(Cheb_Moments[72 + (m + 6*n)*2+(1:2),:], 1), 1)/2)
    plot(Energies, DofS, linewidth=0.5)
end
end

ax[:set_xlim]([-1.2, 1.2]);
ax[:set_ylim]([0, 1.5]);

ax = subplot(122)
Energies, DofS = DoS(E_rescale, 2, squeeze(sum(Cheb_Moments, 1), 1)/n_nodes);
ax[:plot](Energies, DofS, linewidth=0.5)
# ax[:set_xlim]([-1.2, 1.2]);
# ax[:set_ylim]([0, 1.5]);

fig[:canvas][:draw]()