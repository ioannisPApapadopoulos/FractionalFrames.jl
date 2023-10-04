using FractionalFrames, SpecialFunctions, HypergeometricFunctions, LinearAlgebra
using Plots, DelimitedFiles, LaTeXStrings

"""
Section 7.1:

Solving (I + (-Δ)ˢ) u = f, where the solution is u(x) = exp(-x²). 
"""

λ, s = 1, 1/3

# Exact solution
ua = x -> exp(-x^2) 

tfa = (x, s) -> (λ * exp(-x^2) 
            + 2^(2*s*one(eltype(x)))*one(eltype(x))*gamma((s+1/2)*one(eltype(x)))/gamma(one(eltype(x))/2) * _₁F₁((s+1/2)*one(eltype(x)),one(eltype(x))/2,-x^2))

# Right-hand side
ga = (x, s) -> x ≈ zero(eltype(x)) ? ( tfa(-eps(T), s) + tfa(eps(T), s) ) / 2 : tfa(x, s)

# Intervals that contain the centre of the frames
intervals = [-5.,-3,-1,1,3,5]

T = Float64

# Solution approximation space is {T̃, W}
T̃ = ExtendedJacobi{T}(-s,-s)
W = ExtendedWeightedJacobi{T}(s,s)
L = AbsLaplacianPower(axes(W,1), s)

# Want to expand right-hand side in {P, Q}
P = T̃ + L*T̃
Q = W + L*W

# Collocation points
M = 5001; Me = 5001; Mn = 10
xc = collocation_points(M, Me, I=intervals, endpoints=[-10*one(T),10*one(T)], innergap=1e-2)

# Form the frame for the solution
Sₚ = SumSpace{T, Tuple{typeof(T̃), typeof(W)}}((T̃, W), intervals)
# Form the frame of the right-hand side
S = SumSpace{T, Tuple{typeof(P), typeof(Q)}}((P, Q), intervals)


gs = []
rhs_error = []
soln_error = []
Ns = 10:10:250
A = S[xc, 1:Ns[end]]
Aₚ = Sₚ[xc, 1:Ns[end]]

# Run expansion of right-hand side for increasing truncation degree
for N in Ns
    # Expand right-hand side
    g = Matrix(A[:,1:N]) \ ga.(xc, s)
    # Store coefficients
    append!(gs, [g])
    # writedlm("gaussian-coefs.txt", gs)

    # Compute error in right-hand side approximation
    append!(rhs_error, norm(A[:,1:N]*g-ga.(xc, s), Inf))
    # writedlm("rhs_error_gaussian.txt", rhs_error)

    # Compute errer in solution expansion
    append!(soln_error, norm(Aₚ[:,1:N]*g-ua.(xc), Inf))
    # writedlm("soln_error_gaussian.txt", soln_error)

    print("n = $N \n")
end


## Plotting

# Plot norm of coefficients
plot(Ns, norm.(gs,Inf),
    markers=:circle,
    xlabel=L"$\# \, \mathrm{frame \; functions}$",
    ylabel=L"$\infty\mathrm{-norm}$",
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    linewidth=2,
    marker=:dot,
    markersize=3,
    legend =:none
)
savefig("coeff-gaussian-jacobi.pdf")

# Plot convergence of right-hand side and solution
plot(Ns, [rhs_error soln_error],
    label=["RHS" "Solution"],
    ylabel=L"$\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\# \, \mathrm{frame \; functions}$",
    ylim=[1e-15, 5e0],
    yscale=:log10,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    linewidth=2,
    markershape=[:circle :diamond],
    markersize=5,
    yticks=[1e-15, 1e-10,1e-5, 1e0]
)
savefig("error-gaussian-jacobi.pdf")