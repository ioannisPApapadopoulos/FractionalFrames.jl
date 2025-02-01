using FractionalFrames, LinearAlgebra
using Plots, LaTeXStrings

"""
Section 7.5

Solving the fractional heat equation
    (∂ₜ + (-Δ)ˢ) u = 0, u₀(x) = (1+x²)⁻ˢ.

We pick s=1/2 as we know the fundamental solution. We discretize in the time with
Gauss-Legendre(3) (6th order).

We measure the error of the tails at t=1 in the domain x ∈ [-10^3, 10^3].

We see that the spectral method captures the algebraic decay.
"""

# Gauss-Legendre(3)
A = [5/36 2/9-sqrt(15)/15 5/36-sqrt(15)/30;
    5/36+sqrt(15)/24 2/9 5/36-sqrt(15)/24;
    5/36+sqrt(15)/30 2/9+sqrt(15)/15 5/36]
b = [5/18 4/9 5/18]

# Explicit solution
y = (x,t) -> (1 + t) / ((x^2 + (1+t)^2))

# Initial condition
u0 = x -> 1. / (x^2 + 1)

T = Float64
# Approximation functions
T̃ = ExtendedChebyshevT{T}()
W = ExtendedWeightedChebyshevU{T}()
L = AbsLaplacianPower(axes(W,1), 1/2)

# The intervals that the approximation functions are centred on.
intervals = [-5.,-3,-1,1,3,5]

# dual sum space components
P = (L*T̃)[:,2:∞]
Q = L*W

# Collocation points
M = 2000; Mn = 250
xc = collocation_points(M, M, I=intervals, endpoints=[-20*one(T),20*one(T)], innergap=1e-4)

# Sum space approximation the solution
# Evalulation is SIGNIFICANTLY faster if we pass the type of the tuple
Sₚ = SumSpace{T, Tuple{typeof(T̃[:, 2:∞]), typeof(W)}}((T̃[:, 2:∞], W), intervals)
# Least-squares frame matrix for solution space
@time Aₚ = Matrix(Sₚ[xc, 1:Mn]);

# Dual sum space
# Evalulation is SIGNIFICANTLY faster if we pass the type of the tuple
S = SumSpace{T, Tuple{typeof(P), typeof(Q)}}((P, Q), intervals)
Aₛ   = Matrix(S[xc, 1:Mn]);

# Expand initial condition in solution space
u₀ = Aₚ \ u0.(xc)
norm(Aₚ*u₀ - u0.(xc), Inf)
# plot(xc, abs.(Aₚ*u₀ - u0.(xc)))

# λ is the inverse of the timestep δt
λ = 1e2

δt = inv(λ)
# Expansion least-squares matrix as outlined in manuscript
Aₖ = kron(I(length(b)), Aₚ) + δt.*kron(A, Aₛ)

# SVD factorize to save time in solve loops
U, σ, Vs = svd(Aₖ)
tol = 1e1*eps(); filter!(>(tol), σ)
r = length(σ)

timesteps=Int(λ); u = [u₀]

# Run Runge-Kutta time for loop!
for k_ = 1:timesteps
    l = length(b)
    b_ = -Aₛ*u[k_]
    k = Vs[:,1:r] * (inv.(σ) .* (U[:,1:r]' * repeat(b_, l)))
    s = [b[j]*k[(j-1)*Mn+1:j*Mn] for j in 1:l]
    append!(u, [u[k_] + δt.*sum(s)])
end

# Rearrange the evalualtion into matrix form
Us = zeros(length(u[1]), length(0:inv(λ):timesteps*inv(λ)))
for j in 1:lastindex(u)
    Us[:,j] = u[j]
end

xtails = -1000:1000
# Compute pointwise norm
errors = abs.(Sₚ[xtails,1:Mn]*Us[:,end] - y.(xtails, 1))

plot(xtails, errors,
    ylabel=L"$|u(x,1) - \mathbf{\Phi}_N(x)\mathbf{u}_J|$",
    xlabel=L"$x$",
    title=L"$t=1$",
    legend=:none,
    xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    titlefontsize=20,
    linewidth=2,
    markersize=5,
    yaxis=:log10,
    yticks=[1e-17,1e-16,1e-15,1e-14,1e-13,1e-12,1e-11]
)
savefig("fractional-heat-tails.pdf")
