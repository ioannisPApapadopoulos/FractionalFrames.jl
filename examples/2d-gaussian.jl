using FractionalFrames
using StaticArrays, SpecialFunctions, HypergeometricFunctions, LinearAlgebra
using Plots, LaTeXStrings
using Serialization
# include("../plotting/plotting.jl")


# Gaussian Solution
function ua(xy)
    x, y = first(xy), last(xy)
    exp(-x^2-y^2)
end

# RHS corresponding to (-Δ)^(1/2) exp(-x^2-y^2)
function ga(xy, s)
    x, y = first(xy), last(xy)
    # λ * exp(-x^2-y^2) + 4^s * gamma(s+1) *_₁F₁(s+1, 1, -x^2-y^2)
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


# Can consider two kind of collocation points:
#
# (1) pick radial collocation points (avoiding edges of disks where things blow up)
#     then tensor those with angular collocation points "collocation_points_disk".

# (2) Square tensor product of collocation points "collocation_points_square"

function collocation_points_disk()
    M = 1001; Me = 1001;
    r = collocation_points(M, Me, I=vcat(0, js), endpoints=[eps(),10*one(T)], innergap=1e-3)
    r = r[Me+1:end]
    
    # θ = 0:0.05*π:2π
    θ = range(0, 2π, 30) 
    SVector.(r.*cos.(θ)', r.*sin.(θ)')
end

function collocation_points_square()
    SVector.(range(0,10,470), range(0,10,470)')
end

xy = collocation_points_disk()
# xy = collocation_points_square()


# Sum space, interlacing degree, functions, and translations.
Sp = [];S_ = [];
for j in js
    append!(Sp, [SumSpace{Float64, Tuple{typeof(T̃), typeof(W)}}((T̃,W), [-j, j])])
    append!(S_, [SumSpace{Float64, Tuple{typeof((1/j).*P),typeof((1/j).*Q)}}(((1/j).*P,(1/j).*Q), [-j, j])])
end
Sₚ = SumSpace{Float64, NTuple{length(Sp), eltype(Sp)}}(Tuple(Sp), [-1.,1.])
S = SumSpace{Float64, NTuple{length(S_), eltype(S_)}}(Tuple(S_), [-1.,1.])


N = 250 # truncation "degree"
r = range(0,10,100) # for plotting slices


### Use this code to test the Zernike's are doing the correct thing.
@time Aₚ = Sₚ[xy[:], 1:N]     # Assemble least-square matrix
u = Aₚ \ ua.(xy[:])           # Find coefficients for 2D Gaussian
err = abs.(Aₚ*u-ua.(xy[:]))  # Inf-norm error at collocation points
norm(err, Inf)

# Plot slices of how the function looks
Plots.plot(r,ua.(SVector.(r*cos(0.1),r*sin(0.1))))
Plots.plot!(r, Sₚ[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u)

# Plot slice of the error
y = abs.(ua.(SVector.(r*cos(0.1),r*sin(0.1))) - Sₚ[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u)
Plots.plot(r,y)

@time A = S[xy[:], 1:N]        # Assemble least-squares matrix for frame of RHS
err_f = abs.(A*u-ga.(xy[:],s)) # Measure error given the coefficients we just computed
norm(err_f, Inf)

# Plot slices
Plots.plot(r,ga.(SVector.(r*cos(0.1),r*sin(0.1)),s))
Plots.plot!(r, S[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u)


y = abs.(ga.(SVector.(r*cos(0.1),r*sin(0.1)),s) - S[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u)
Plots.plot(r,y)

using DelimitedFiles
writedlm("errors.log", errors)
writedlm("nrm_cfs.log", nrm_cfs)

mode_0(n) = 1 + sum(1:2*n)

#### Solve
A = deserialize("A")
Aₚ = deserialize("Ap")
errors = []
nrm_cfs = []
# for N in 10:10:1000

for N in 1380:10:1720
    u = A[:,1:N] \ ga.(xy[:], s)
    err = abs.(Aₚ[:,1:N]*u-ua.(xy[:]))  # Inf-norm error at collocation points
    append!(errors, [norm(err, Inf)])
    append!(nrm_cfs, [norm(u, Inf)])
    print("Completed n = $N.\n")
end


errors = readdlm("errors.log")
nrm_cfs = readdlm("nrm_cfs.log")


Plots.plot(10:10:90, errors[mode_0.(0:8)],
    # label=L"$\mathrm{Chebyshev} \otimes \mathrm{Fourier}$",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,
    # legend=:bottomleft,
    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \, \mathrm{frame \; functions}$",
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    gridlinewidth = 2,
    legend=:none,
    # yticks=[1e-15,1e-10,1e-5,1e-3,1e-1], # 1e-5
    ylim=[1e-9,1e0], # 1e-5
    xlim=[0, 95],
    xticks=0:10:90
)
Plots.savefig("convergence-2d-gaussian.pdf")

Plots.plot(10:10:90, nrm_cfs[mode_0.(0:8)],
    # label=L"$\mathrm{Chebyshev} \otimes \mathrm{Fourier}$",
    linewidth=2,
    markershape=:dtriangle,
    markersize=5,
    # legend=:bottomleft,
    ylabel=L"$\ell^\infty\mathrm{-norm}$",
    xlabel=L"$\# \, \mathrm{frame \; functions}$",
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    gridlinewidth = 2,
    legend=:none,
    xlim=[0, 95],
    xticks=0:10:90
    # yticks=[1e-15,1e-10,1e-5,1e-3,1e-1], # 1e-5
    # ylim=[1e-9,1e0] # 1e-5
)
Plots.savefig("cfs-2d-gaussian.pdf")


N = 1360
@time A = S[xy[:], 1:N]
u2 = A \ ga.(xy[:], s)
err_f = abs.(A*u2-ga.(xy[:],s)) # Measure error given the coefficients we just computed
norm(err_f, Inf)
y = abs.(ga.(SVector.(r*cos(0.1),r*sin(0.1)),s) - S[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u2)
Plots.plot(r,y)

@time As = S[xy[:], 1371:1720]
@time As2 = Sₚ[xy[:], 1371:1720]  

A = hcat(A, As);
Aₚ = hcat(Aₚ, As2)


@time Aₚ = Sₚ[xy[:], 1:N]     # Assemble least-square matrix       # Find coefficients for 2D Gaussian
err = abs.(Aₚ*u2-ua.(xy[:]))  # Inf-norm error at collocation points
norm(err, Inf)
y = abs.(ua.(SVector.(r*cos(0.1),r*sin(0.1))) - Sₚ[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u2)
Plots.plot(r,y)
# Plots.plot(r,ga.(SVector.(r*cos(0.1),r*sin(0.1)),s))
# Plots.plot!(r, S[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u2)
# y = abs.(ga.(SVector.(r*cos(0.1),r*sin(0.1)),s) - S[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u2)
# Plots.plot(r,y)


serialize("Ap", Aₚ)
serialize("A", A)

Af = A;
Aₚf = Aₚ;
# @time Aₚ = Sₚ[xy[:], 1:N]
# err = abs.(Aₚ*u2-ua.(xy[:]))
# norm(err, Inf)
# frame_plot(θ,r,Aₚ*u2)


# Plots.plot(r,ua.(SVector.(r*cos(0.1),r*sin(0.1))))
# Plots.plot!(r, Sₚ[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u2)
# y = abs.(ua.(SVector.(r*cos(0.1),r*sin(0.1))) - Sₚ[SVector.(r*cos(0.1),r*sin(0.1)), 1:N]*u2)
# Plots.plot(r,y)