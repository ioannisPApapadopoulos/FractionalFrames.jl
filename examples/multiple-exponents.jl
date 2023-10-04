using FractionalFrames, SpecialFunctions, HypergeometricFunctions
using LinearAlgebra
using Plots

λ, s, t =1, 1/3, 1/5

# Exact solution
ua = x -> exp(-x^2) 

tfa = (x, s, t) -> (λ * exp(-x^2) 
            + 2^(2*s*one(eltype(x)))*one(eltype(x))*gamma((s+1/2)*one(eltype(x)))/gamma(one(eltype(x))/2) * _₁F₁((s+1/2)*one(eltype(x)),one(eltype(x))/2,-x^2)
            + 2^(2*t*one(eltype(x)))*one(eltype(x))*gamma((t+1/2)*one(eltype(x)))/gamma(one(eltype(x))/2) * _₁F₁((t+1/2)*one(eltype(x)),one(eltype(x))/2,-x^2)
            )

# Right-hand side
ga = (x, s, t) -> x ≈ zero(eltype(x)) ? ( tfa(-eps(T), s, t) + tfa(eps(T), s, t) ) / 2 : tfa(x, s, t)

# Intervals that contain the centre of the frames
intervals = [-5.,-3,-1,1,3,5]

T = Float64

# Solution approximation space is {T̃, W}
T̃ = GeneralExtendedJacobi{T}(-1/4, -1/4)
W = ExtendedWeightedJacobi{T}(1/4, 1/4)
Ls = AbsLaplacianPower(axes(W,1), s)
Lt = AbsLaplacianPower(axes(W,1), t)

# Want to expand right-hand side in {P, Q}
P = T̃ + Ls*T̃ + Lt*T̃
Q = W + Ls*W + Lt*W


# Collocation points
M = 5001; Me = 5001;
xc = collocation_points(M, Me, I=intervals, endpoints=[-20*one(T),20*one(T)], innergap=1e-2)

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
    g = Matrix(A[:,1:N]) \ ga.(xc, s, t)
    # Store coefficients
    append!(gs, [g])
    # writedlm("multiple-exponents/logs/gaussian-coefs.txt", gs)

    # Compute error in right-hand side approximation
    append!(rhs_error, norm(A[:,1:N]*g-ga.(xc, s, t), Inf))
    # writedlm("multiple-exponents/logs/rhs_error_gaussian.txt", rhs_error)

    # Compute errer in solution expansion
    append!(soln_error, norm(Aₚ[:,1:N]*g-ua.(xc), Inf))
    # writedlm("multiple-exponents/logs/soln_error_gaussian.txt", soln_error)

    print("n = $N \n")
end


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
savefig("coeff-mult-exp-jacobi.pdf")


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
Plots.savefig("error-mult-exp-jacobi.pdf")

