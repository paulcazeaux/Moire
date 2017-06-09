module TB1
export DoS, Image, GradientImage, Read_DoS

using PyCall
using PyPlot
using Interpolations

function DoS(a, b, mu)
    #mu = mu[1:1024,:];
    N = size(mu,1)
    nratios = size(mu,2)
    M = 2*N

    g = ((N-(0:N-1)).*cos(pi*(0:N-1)/N) + sin(pi*(0:N-1)/N)*cot(pi/N))/N
    g = [g[1]/sqrt(2); g[2:end]]
    Y = cos(pi/M*(0.5:M))
    Y = repmat(Y, 1, nratios)
    Z = zeros(Float64, M, nratios)

    for p=1:nratios
      Z[1:N,p] =  g .* mu[:,p]
      Z[:,p] = idct!(Z[:,p])
    end

    Z = 4*flipdim(Z./(pi*sqrt(1-Y.^2)), 1)
    Y = flipdim(a*Y+b, 1)

    return Y, Z
end


function Image(grid, values, npx, upscaling, xrange, yrange, cscale, crange, xLabel, yLabel, cLabel, number)
  nratios = size(grid, 2)
  if xrange == :custom_ratios
    xmin = .25
    xmax = .75
  else
    xmin = xrange[1]
    xmax = xrange[2]
  end
  ymin = yrange[1]
  ymax = yrange[2]

  Img = zeros(Float64, npx, nratios)

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

  if cscale == :auto
    crange = collect(extrema(Img))
  elseif cscale == :relative
    crange = [crange[1]*minimum(Img), crange[2]*maximum(Img)]
  elseif cscale == :absolute
    crange = collect(crange)
  end

  Img = max(crange[1], Img)
  Img = min(crange[2], Img)
  close(number)
  fig = figure(number, figsize = (15, 10))
  ax = axes()
  img = ax[:imshow](Img, interpolation = "bicubic", origin = "lower",
   extent = (xmin, xmax, ymin, ymax), aspect = "auto", cmap="magma", clim = crange)


  xlabel(xLabel, fontsize=20)
  ylabel(yLabel, fontsize=20)
  ax[:set_xlim]([xmin, xmax])
  ax[:set_ylim]([ymin, ymax])

  
  xtic = [1/9 1/8 1/7 1/6 1/5 1/4 1/3 2/5 1/2 3/5 2/3 4/5 1 5/4 3/2 5/3 2 5/2 3 4 5 6 7 8 9]'
  xtic = sqrt(xtic)./(1+sqrt(xtic))
  xticstr = (L"\frac{1}{9}", L"\frac{1}{8}", L"\frac{1}{7}", 
      L"\frac{1}{6}", L"\frac{1}{5}", L"\frac{1}{4}", L"\frac{1}{3}",
      L"\frac{2}{5}",  L"\frac{1}{2}", L"\frac{3}{5}", L"\frac{2}{3}", L"\frac{4}{5}", L"1",
      L"\frac{5}{4}", L"\frac{3}{2}", L"\frac{5}{3}", L"2", L"\frac{5}{2}", 
      L"3", L"4", L"5", L"6", L"7", L"8", L"9")

  I = find((xtic .>= xmin) & (xtic .<= xmax))
    
  ax[:set_xticks](xtic[I])
  ax[:set_xticklabels](xticstr[I])

  tick_params(axis="x", labelsize=15)
  tick_params(axis="y", labelsize=15)
  ax[:spines]["top"][:set_color]("none")
  ax[:spines]["right"][:set_color]("none")
  # cbar = colorbar(img, ax=ax, orientation=:vertical, shrink=.75,label=cLabel, format="%g")

  # cticks = collect(linspace(crange[1], crange[2], 9))
  # cticklabels = [@sprintf("\$%.1g\$", tick) for tick in cticks]
  # if crange[2] < maximum(Img)
  #   cticklabels[end] = @sprintf("\$\\geq %.1g\$", cticks[end])
  # end
  # cbar[:set_ticks](cticks)
  # cbar[:set_ticklabels](cticklabels)
  
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

function Read_DoS(filename)
  stream = open(filename)
  n = convert(Int, read(stream, Int32))
  nratios = convert(Int, read(stream, Int32))
  S = read(stream, Float64, (n, nratios))
  close(stream)
  return n, nratios, S
end


end
