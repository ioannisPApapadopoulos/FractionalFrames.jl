using FractionalFrames
using StaticArrays, SpecialFunctions, HypergeometricFunctions, LinearAlgebra
using Plots, LaTeXStrings

"""
Section 7.3:

Solving (I + (-Δ)^1/2) u(x,y) = f(x,y), where the solution is u(x,y) = exp(-x²-y²).
"""

# Gaussian Solution
function ua(xy)
    x, y = first(xy), last(xy)
    exp(-x^2-y^2)
end

# RHS corresponding to (-Δ)^(1/2) exp(-x^2-y^2)
function ga(xy, s)
    x, y = first(xy), last(xy)
    4^s * gamma(s+1) *_₁F₁(s+1, 1, -x^2-y^2)
end

T = Float64
s = 1/2;
T̃ = ExtendedZernike(0.0, -s)
W = ExtendedWeightedZernike(0.0, s)

L = AbsLaplacianPower(axes(W,1), s)
P = L*T̃
Q = L*W

# These are going to be the outer radii of the disks we consider
js = [1, 3/2, 2, 3, 4]

# Sum space, interlacing degree, functions, and translations.
Sp = [];S_ = [];
for j in js
    append!(Sp, [SumSpace{Float64, Tuple{typeof(T̃), typeof(W)}}((T̃,W), [-j, j])])
    append!(S_, [SumSpace{Float64, Tuple{typeof((1/j).*P),typeof((1/j).*Q)}}(((1/j).*P,(1/j).*Q), [-j, j])])
end
Sₚ = SumSpace{Float64, NTuple{length(Sp), eltype(Sp)}}(Tuple(Sp), [-1.,1.])
S = SumSpace{Float64, NTuple{length(S_), eltype(S_)}}(Tuple(S_), [-1.,1.])

# Can consider two kind of collocation points:
#
# (1) pick radial collocation points (avoiding edges of disks where things blow up)
#     then tensor those with angular collocation points "collocation_points_disk".

# (2) Square tensor product of collocation points "collocation_points_square"

function collocation_points_disk(M, Me)
    r = collocation_points(M, Me, I=vcat(0, js), endpoints=[eps(),10*one(T)], innergap=1e-3)
    r = r[Me+1:end]

    θ = range(0, 2π, 5) 
    SVector.(r.*cos.(θ)', r.*sin.(θ)')
end

function collocation_points_square()
    SVector.(range(0,10,470), range(0,10,470)')
end

# Since solution is radially symmetric, I only want
# to evaluate columns with Fourier mode (0,0)
mode_0(n) = 1 + sum(1:2*n)
function zero_mode_columns(N::Int, perN::Int)
    a = mode_0.(0:N÷perN-1) .* perN
    b = [a[i]-perN+1:a[i] for i in 1:length(a)]
    vcat(b...)
end

#### Solve
errors = []
nrm_cfs = []

for N in 10:10:110
    xy = collocation_points_disk(N,N) # scale collocation points linearly
    cols = zero_mode_columns(N, 10) # extract zero-Fourier mode columns
    Aₚ = Sₚ[xy[:], cols]    # Assemble least-square matrix for frame of solution
    A = S[xy[:], cols]     # Assemble least-squares matrix for frame of RHS

    u = A[:,1:N] \ ga.(xy[:], s)
    err = abs.(Aₚ[:,1:N]*u-ua.(xy[:]))  # Inf-norm error at collocation points
    append!(errors, [norm(err, Inf)])
    append!(nrm_cfs, [norm(u, Inf)])
    print("Completed n = $N.\n")
end


Plots.plot(10:10:110, errors,
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,
    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \, \mathrm{frame \; functions} \,\,\, (N)$",
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    gridlinewidth = 2,
    legend=:none,
    yticks=[1e-9,1e-6,1e-3,1e0],
    xticks=0:10:110
)
Plots.savefig("convergence-2d-gaussian.pdf")

Plots.plot(10:10:110, nrm_cfs,
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,
    ylabel=L"$\ell^\infty\mathrm{-norm}$",
    xlabel=L"$\# \, \mathrm{frame \; functions} \,\,\, (N)$",
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    gridlinewidth = 2,
    legend=:none,
    # xlim=[0, 95],
    xticks=0:10:110
)
Plots.savefig("cfs-2d-gaussian.pdf")