module TB1
export Hamiltonian, KPM, Occupancy, Conductivity, Conductivity2, Image, GradientImage, Read

using PyPlot
using Interpolations


function Hamiltonian(p,q,W,sigma)

  L1 = (0:p-1)*sqrt(q/p)
  L2 = (0:q-1)*sqrt(p/q)


  H11 = spdiagm((ones(p-1,1), ones(1,1)), (1, (p-1)), p, p)
  H22 = spdiagm((ones(q-1,1), ones(1,1)), (1, (q-1)), q, q)

  I = zeros(Int64, p+ceil(Int64, sqrt(p*q)*12*sigma))
  J = zeros(Int64, p+ceil(Int64, sqrt(p*q)*12*sigma))
  V = zeros(Float64, p+ceil(Int64, sqrt(p*q)*12*sigma))

  n = 1

  for i=1:p
    A = ceil(Int64, sqrt(q/p)*(L1[i] - 6*sigma))
    B = floor(Int64, sqrt(q/p)*(L1[i] + 6*sigma))
    for j = mod(A:B,q)+1
        I[n] = i
        J[n] = j
        d = L1[i] - L2[j]
        d = d - round(d/sqrt(p*q))*sqrt(p*q)
        V[n] = exp(-.5*(d/sigma)^2)
        n = n+1
    end
  end
  I = I[1:n-1]
  J = J[1:n-1]
  V = V[1:n-1]
  H12 = sparse(I,J,V,p,q)
  H21 = spzeros(Float64, q,p)
  H = [H11 W*H12; H21 H22]  # Hermitian stored in upper triangle


  I,J,V = findnz(H);
  K = (I.!=J);
  I = I[K]; J = J[K]; V = V[K];
  L = [L1;L2];
  D = (L[I] - L[J]);
  D = D - round(D/sqrt(p*q))*sqrt(p*q);
  V = D.*V;

  dH = sparse(I,J,V,p+q,p+q) # Hermitian stored in upper triangle
  return H, dH
end

function cheb_recursion_shared!(D)
  M = size(D,1)
  N = size(D,2)-1
  nwks = length(workers())
  idx = indexpids(D)
  rge = 1+div((idx-1) * M, nwks) : div(idx * M, nwks)

  for n=2:N
      D[rge,n+1] = 2*D[rge,2].*D[rge,n] - D[rge,n-1];
  end
end

function KPM(p,q,W,sigma,N)
  BLAS.set_num_threads(4)
  H, dH = Hamiltonian(p,q,W,sigma)
  e = eigs(H+H', nev = 2, which = :BE, tol = 1e-2, ncv = 20, ritzvec = false)[1];

  H_f = full(H);
  dH_f = full(dH);
  d,v = LAPACK.syevr!('V', 'A', 'U', H_f, 0.0, 0.0, 0, 0, -1.0)
  a = (d[end]-d[1])/(2 - .2)
  b = (d[end]+d[1])/2
  d = (d - b)/a

  D = SharedArray(Float64, (p+q, N+1));
  D[:,1] = 1
  D[:,2] = d

  @sync begin
    for pw in workers()
      @async remotecall_wait(cheb_recursion_shared!, pw, D)
    end
  end

  J = v'*(dH-dH')*v
  mu = -D.'*(J.*J.')*D
  d = d*a+b
  return d, mu, a, b
end



function Occupancy(Ef, d, beta)
  S = zeros(Float64, size(Ef))
  for i=1:size(Ef)[1]
    S[i] = sum(1./(1. + exp(beta*(d - Ef[i]))))
  end
  S = S / size(d)[1]
  return S
end

function Conductivity(a, b, mu, beta, tau,  E1, E2, n)
    N = size(mu,1)

    E = b+a*cos(pi/N*(0.5:N))
    maxE = max(E.+0*E', 0*E.+E')
    minE = min(E.+0*E', 0*E.+E')

    F0 = (expm1(-beta*(maxE - minE))) ./ (minE - maxE)
    F0[isnan(F0)] = beta

    g = ((N-(0:N-1)).*cos(pi*(0:N-1)/N) + sin(pi*(0:N-1)/N)*cot(pi/N))/N
    g = [g[1]/sqrt(2); g[2:end]]
    V = (g*g').*mu;
    idct!(V)
    V .*= F0
    V ./= (1/tau + tau*(E' .- E).^2)

    Ef = linspace(E1, E2, n)
    S = SharedArray(Float64, n)
    @sync @parallel for i=1:n
       S[i] = sum(V ./ (1 + exp(-beta*(maxE-Ef[i]))) ./ (1 + exp(beta*(minE-Ef[i]))))
    end
    S .*= (a/N)^2
    return Ef, S
end

function Conductivity2(a,b,mu, beta, tau,  E1, E2, n1, omega1, omega2, n2)
    N = size(mu,1)

    E = b+a*cos(pi/N*(0.5:N))
    maxE = max(E.+0*E', 0*E.+E')
    minE = min(E.+0*E', 0*E.+E')

    F0 = (expm1(-beta*(maxE - minE))) ./ (minE - maxE)
    F0[isnan(F0)] = beta

    g = ((N-(0:N-1)).*cos(pi*(0:N-1)/N) + sin(pi*(0:N-1)/N)*cot(pi/N))/N
    g = [g[1]/sqrt(2); g[2:end]]
    V = (g*g').*mu;
    idct!(V)
    V .*= F0

    Ef = linspace(E1, E2, n1)
    omega = linspace(omega1, omega2, n2)
    S = SharedArray(Complex128, n1, n2)
    @sync @parallel for j=1:n2
      W = V ./ (1/tau + 1im*(E' .- E - omega[j]))
      for i=1:n1
         S[i,j] = sum(W ./ (1 + exp(-beta*(maxE-Ef[i]))) ./ (1 + exp(beta*(minE-Ef[i]))))
      end
    end

    S .*= (a/N)^2
    return Ef, omega, S
end

function Image(grid, values, npx, upscaling, xrange, yrange, crange, xLabel, yLabel, cLabel, number)
  nratios = size(grid, 2)
  if xrange == :custom_ratios
    xmin = 1/7
    xmax = 6/7
  else
    xmin = xrange[1]
    xmax = xrange[2]
  end
  ymin = yrange[1]
  ymax = yrange[2]

  Img = Array(Float64, npx, nratios)

  for p=1:nratios
    len = size(values, 1)
    val = grid[:,p]
    if upscaling != 1
      itp = interpolate(val, BSpline(Cubic(Line())), OnGrid())
      val = sort!(itp[linspace(1, len, upscaling*len)])
    end

    itp = interpolate((val, ), linspace(1, len, upscaling*len), Gridded(Linear()))
    idx = itp[linspace(ymin, ymax, npx)]
    itpc = extrapolate(interpolate(values[:,p], BSpline(Cubic(Line())), OnGrid()), 0.)
    for j=1:npx
      Img[j,p] = itpc[idx[j]]
    end
  end

  close(number)
  fig = figure(number, figsize = (15, 10))
  ax = axes()
  img = ax[:imshow](Img, interpolation = "bicubic", origin = "lower",
   extent = (xmin, xmax, ymin, ymax), aspect = "auto", cmap="magma", clim = collect(crange))

  xlabel(xLabel, fontsize=20)
  ylabel(yLabel, fontsize=20)
  ax[:set_xlim]([xmin, xmax])
  ax[:set_ylim]([ymin, ymax])

  if xrange == :custom_ratios
    xtic = [1/6 1/5 1/4 1/3 2/5 1/2 3/5 2/3 4/5 1 5/4 3/2 5/3 2 5/2 3 4 5 6]'
    xtic = xtic./(1+xtic)
    xticstr = (L"\frac{1}{6}", L"\frac{1}{5}", L"\frac{1}{4}", L"\frac{1}{3}",
      L"\frac{2}{5}",  L"\frac{1}{2}", L"\frac{3}{5}", L"\frac{2}{3}", L"\frac{4}{5}", L"1",
      L"\frac{5}{4}", L"\frac{3}{2}", L"\frac{5}{3}", L"2", L"\frac{5}{2}", L"3", L"4", L"5", L"6")
    ax[:set_xticks](xtic)
    ax[:set_xticklabels](xticstr)
    tick_params(axis="x", labelsize=20)
  else
    tick_params(axis="x", labelsize=15)
  end
  tick_params(axis="y", labelsize=15)
  ax[:spines]["top"][:set_color]("none")
  ax[:spines]["right"][:set_color]("none")
  cbar = colorbar(img, ax=ax, orientation=:vertical, shrink=.75,label=cLabel, format="%g")
  return fig
end

function GradientImage(grid, values, npx, upscaling, xrange, yrange, cscale, crange, xLabel, yLabel, cLabel, number)
  nratios = size(grid, 2)
  if xrange == :custom_ratios
    xmin = 1/7
    xmax = 6/7
  else
    xmin = xrange[1]
    xmax = xrange[2]
  end
  ymin = yrange[1]
  ymax = yrange[2]
  Img = Array(Float64, npx, nratios)

  for p=1:nratios
    len = size(values, 1)
    val = grid[:,p]
    if upscaling != 1
      itp = interpolate(val, BSpline(Cubic(Line())), OnGrid())
      val = sort!(itp[linspace(1, len, upscaling*len)])
    end

    itp = interpolate((val, ), linspace(1, len, upscaling*len), Gridded(Linear()))
    idx = itp[linspace(ymin, ymax, npx)]
    itpc = interpolate(values[:,p], BSpline(Cubic(Line())), OnGrid())
    for j=1:npx
      if idx[j] >= 1 && idx[j] <= len
        Img[j,p] = gradient(itpc, idx[j])[1]
      else
        Img[j,p] = 0.
      end
    end
  end

  if cscale == :auto
    crange = collect(extrema(Img))
  elseif cscale == :relative
    crange = [crange[1]*minimum(Img), crange[2]*maximum(Img)]
  elseif cscale == :absolute
    crange = collect(crange)
  end

  close(number)
  fig = figure(number, figsize = (15, 10))
  ax = axes()
  img = ax[:imshow](Img, interpolation = "hermite", origin = "lower",
   extent = (xmin, xmax, ymin, ymax), aspect = "auto", cmap="magma", clim = crange)

  xlabel(xLabel, fontsize=20)
  ylabel(yLabel, fontsize=20)
  ax[:set_xlim]([xmin, xmax])
  ax[:set_ylim]([ymin, ymax])

  if xrange == :custom_ratios
    xtic = [1/6 1/5 1/4 1/3 2/5 1/2 3/5 2/3 4/5 1 5/4 3/2 5/3 2 5/2 3 4 5 6]'
    xtic = xtic./(1+xtic)
    xticstr = (L"\frac{1}{6}", L"\frac{1}{5}", L"\frac{1}{4}", L"\frac{1}{3}",
      L"\frac{2}{5}",  L"\frac{1}{2}", L"\frac{3}{5}", L"\frac{2}{3}", L"\frac{4}{5}", L"1",
      L"\frac{5}{4}", L"\frac{3}{2}", L"\frac{5}{3}", L"2", L"\frac{5}{2}", L"3", L"4", L"5", L"6")
    ax[:set_xticks](xtic)
    ax[:set_xticklabels](xticstr)
    tick_params(axis="x", labelsize=20)
  else
    tick_params(axis="x", labelsize=15)
  end
  tick_params(axis="y", labelsize=15)
  ax[:spines]["top"][:set_color]("none")
  ax[:spines]["right"][:set_color]("none")
  cbar = colorbar(img, ax=ax, orientation=:vertical, shrink=.75, label=cLabel)

  cticks = collect(linspace(crange[1], crange[2], 9))
  cticklabels = [@sprintf("\$%.1g\$", tick) for tick in cticks]
  if crange[2] < maximum(Img)
    cticklabels[end] = @sprintf("\$\\geq %.1g\$", cticks[end])
  end
  cbar[:set_ticks](cticks)
  cbar[:set_ticklabels](cticklabels)

  return fig
end

function Read(filename)
  stream = open(filename)
  n = read(stream, Int)
  nratios = read(stream, Int)
  S = read(stream, Float64, (n, nratios, 3))
  close(stream)
  return n, nratios, S[:,:,1], S[:,:,2], S[:,:,3]
end
end
