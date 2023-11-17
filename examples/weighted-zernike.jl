using FractionalFrames, MultivariateOrthogonalPolynomials
using StaticArrays, PyPlot, LaTeXStrings

"""
In this example we how knowledge of the explicit expressions may be used in a
spectral method for solving (-Δ)^(1/3) u(x,y) = f(x,y) on the whole space ℝ².

We consider a right-hand side that is supported on the unit disk but has a blowup
as r→1.
"""

# right-hand side in polar coordinates
f_polar(r,θ) = r < 1 ? 20*(1-r^2)^(-1/3) * r^3*cos(3θ)*exp(-r^2) : 0
function f(x,y)
    r = sqrt(x^2 + y^2)
    θ = atan(y, x)
    f_polar(r,θ)
end

# Weighted Zernike basis
wZ = Weighted(Zernike(-1/3))

x,y = first.(axes(wZ,1)), last.(axes(wZ,1))
# Coefficients for RHS expanded in weighted Zernike
fc = wZ \ f.(x,y)

# Find the indices that that are nonzero. This vastly speeds up
# later computations
inds = findall(x->abs.(x)>1e-14, fc[Block.(1:200)])

wZ = ExtendedWeightedZernike(0.0,-1/3)
# Extended Zernike basis
Z̃ = ExtendedZernike(0.0, -1/3)

xx = -2:0.01:2
xy = SVector.(xx,xx')

# Evaluate RHS
F = reshape(wZ[xy[:], inds]*fc[inds], size(xy,1), size(xy,2))'
# Evaluate solution
U = reshape(Z̃[xy[:], inds]*fc[inds], size(xy,1), size(xy,2))'

# Helper plotting function
function plotZ(U, ttl, tt)
    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true

    fig = plt.figure()
    ax = fig.add_subplot(111)
    vmin, vmax = minimum(U), maximum(U)
    pc = pcolormesh(xx, xx, U, norm=PyPlot.matplotlib.colors.SymLogNorm(linthresh=1, linscale=1,vmin=vmin, vmax=vmax, base=10), cmap="bwr", shading="gouraud")
    cbar = plt.colorbar(pc)
    cbar.set_label(ttl)
    cbar.set_ticks(tt)
    cbar.set_ticklabels(tt)
    ax.set_xlabel(latexstring(L"$x$"))
    ax.set_ylabel(latexstring(L"$y$"))
    ax.set_aspect("equal")
    ax.set_yticks(-2:2)
    display(gcf())
end

# Plot solutions
plotZ(F, latexstring(L"$f(x,y)$"), -0.1:0.04:0.1)
PyPlot.savefig("rhs.png",dpi=500)
plotZ(U, latexstring(L"$u(x,y)$"), [-4,-1,-0.5,0,0.5,1,4])
PyPlot.savefig("sol.png",dpi=500)