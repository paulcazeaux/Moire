include("TB1.jl")
using TB1
using PyPlot
pygui(true)

n, n_nodes, Cheb_Moments = Read_LDoS("bi_graphene_0.7.jld")
E_rescale = 11.35;


fig = figure(1)
clf()
ax=axes()
# ax = subplot(211)

N = 12;

C = ColorMap("jet")
f(x,y) = C(exp(-2*((x+y/2)^2 + 3/4*y^2)) )

for m=0:N-1
for n=0:N-1
    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[(m + 6*n)*2+1, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[(m + 6*n)*2+2, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    Energies, DofS = DoS(E_rescale, 2, squeeze(sum(Cheb_Moments[(m + N*n)*2+(1:2),1:2000], 1), 1)/2)
    plot(Energies, DofS, linewidth=1, color=f(1-2*m/N,1-2*n/N))

    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[72+(m + 6*n)*2+1, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    # Energies, DofS = DoS(E_rescale, 2, Cheb_Moments[72+(m + 6*n)*2+2, :])
    # ax[:plot](Energies, DofS, linewidth=0.5)
    Energies, DofS = DoS(E_rescale, 2, squeeze(sum(Cheb_Moments[2*N^2 + (m + N*n)*2+(1:2),1:2000], 1), 1)/2)
    plot(Energies, DofS, linewidth=1, color=f(1-2*m/N,1-2*n/N))
end
end

ax[:set_xlim]([-1, 1]);
ax[:set_ylim]([0, 2.5]);

# ax = subplot(212)
# Energies, DofS = DoS(E_rescale, 2, squeeze(sum(Cheb_Moments, 1), 1)/n_nodes);
# ax[:plot](Energies, DofS, linewidth=2, color="blue")
# ax[:set_xlim]([-1, 1]);
# ax[:set_ylim]([0, 1]);

xlabel("Energy")
ylabel("Density of States")

fig[:canvas][:draw]()

savefig("bi_graphene_LDoS_0.7.pdf")