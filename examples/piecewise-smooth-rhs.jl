using FractionalFrames, SpecialFunctions, HypergeometricFunctions, LinearAlgebra
using Plots, DelimitedFiles, LaTeXStrings

"""
Section 7.3:

Solving (I+(-Δ)^{1/5}) u = f where f is piecewise smooth.
"""

s = 1/5

function fa(x)
    if x < 0
        return 1/(1+x^2)
    else
        return exp(-x)
    end
end

# Intervals that contain the centre of the frames
intervals = range(-16, 16, 9)

T = Float64

# Solution approximation space is {T̃, W}
T̃ = ExtendedJacobi{T}(-s,-s)
W = ExtendedWeightedJacobi{T}(s,s)
L = AbsLaplacianPower(axes(W,1), s)

# Want to expand right-hand side in {P, Q}
P = T̃ + L*T̃
Q = W + L*W

# Form the frame for the solution
Sₚ = SumSpace{T, Tuple{typeof(T̃), typeof(W)}}((T̃, W), intervals)
# Form the frame of the right-hand side
S = SumSpace{T, Tuple{typeof(P), typeof(Q)}}((P, Q), intervals)

gs = []
rhs_error = []
rhs_cauchy_error = []
soln_error = []


Ns = 10:10:350
# Run expansion of right-hand side for increasing truncation degree
for N in Ns
    xc = collocation_points(N, N, I=intervals, endpoints=[-25*one(T),25*one(T)], innergap=1e-2)

    A = Matrix(S[xc, 1:N])
    Aₚ = Matrix(Sₚ[xc, 1:N])

    # Expand right-hand side
    g = A \ fa.(xc)
    # Store coefficients
    push!(gs, g)

    # Compute error in right-hand side approximation
    push!(rhs_error, norm(A*g-fa.(xc), Inf))

    # Compute Cauchy errors
    if N > 10
        push!(rhs_cauchy_error, norm(A*g-Matrix(S[xc, 1:N-10])*gs[end-1], Inf))
        push!(soln_error, norm(Aₚ*g-Matrix(Sₚ[xc, 1:N-10])*gs[end-1], Inf))
    end

    print("n = $N \n")
end


## Plotting

# Plot norm of coefficients
plot(Ns, norm.(gs,Inf),
    markers=:circle,
    xlabel=L"$\# \, \mathrm{frame \; functions} \,\,\, (N)$",
    ylabel=L"$\ell^\infty\mathrm{-norm}$",
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    linewidth=2,
    marker=:dot,
    markersize=3,
    legend =:none,
    yscale=:log10,
    yticks=[1e-1,1e0,1e1,1e2,1e3,1e4,1e5]
    # ylim=[0,1e4]
)
savefig("coeff-piecewise-smooth.pdf")

# Plot convergence of right-hand side and solution
plot(Ns[2:end], [rhs_error[2:end] rhs_cauchy_error soln_error],
    label=["RHS (exact)" "RHS (Cauchy)" "Solution (Cauchy)"],
    ylabel=L"$\mathrm{(Cauchy)} \;\; \ell^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \, \mathrm{frame \; functions}\,\,\, (N)$",
    ylim=[8e-10, 5e0],
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    linewidth=2,
    markershape=[:circle :square :diamond],
    color=[theme_palette(:auto)[1] theme_palette(:auto)[3] theme_palette(:auto)[2]],
    markersize=5,
    yticks=[1e-9, 1e-7, 1e-5, 1e-3, 1e-1, 1e1]
)
savefig("error-piecewise-smooth.pdf")