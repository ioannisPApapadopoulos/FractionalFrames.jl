using FractionalFrames, LinearAlgebra
using Plots, LaTeXStrings

"""
Section 7.3

Solving a fractional heat equation with a variable exponent s(t)
    (∂ₜ + (-Δ)ˢ⁽ᵗ⁾) u = 0, u₀(x) = (1+x²)⁻ˢ.

We discretize in time using an implicit midpoint rule (second order).
"""

# Implicit midpoint rule Butcher tableau
A = [1/2]
b = [1.]

T = Float64

# Intervals to centre the approximation functions on
intervals = [-5.,-3,-1,1,3,5]

# Initial state
u0 = x -> 1. / (x^2 + 1) 

# Collocation points
M = 5001; Me = 5001; Mn = 250
xc = collocation_points(M, Me, I=intervals, endpoints=[-20*one(T),20*one(T)], innergap=1e-4)

# Variable exponent of fractional Laplacian
frac_s(t) = 1/2 - t/3

# Time-steps
λ = 100
δt = inv(λ)

# For each time-step we are required to adjust the choice of Jacobi parameters.
# So here we initialise all the required functions and store them in a list.
Sₚ = []; S = []
for k_ = 0:λ
    t = δt * k_
    T̃ = ExtendedJacobi{T}(-frac_s(t),-frac_s(t))
    W = ExtendedWeightedJacobi{T}(frac_s(t),frac_s(t))
    L = AbsLaplacianPower(axes(T̃, 1), frac_s(t))
    P = L * T̃
    Q = L * W
    # Evalulation is SIGNIFICANTLY faster if we pass the type of the tuple
    append!(Sₚ, [SumSpace{T, Tuple{typeof(T̃), typeof(W)}}((T̃, W), intervals)])
    append!(S, [SumSpace{T, Tuple{typeof(P), typeof(Q)}}((P, Q), intervals)])
end

Aₚ_ = Sₚ[1][xc, 1:Mn]
# Expand initial condition
u₀ = Matrix(Aₚ_) \ u0.(xc)
As_ = S[1][xc, 1:Mn]

u = [u₀]
# Run time-stepping loop
@time for k_ = 1:λ
    # Assemble least-squares frame matrices
    Aₛ = Matrix(S[k_+1][xc, 1:Mn]);
    Aₚ = Matrix(Sₚ[k_+1][xc, 1:Mn]);
    # Runge-Kutta least squares frame matrix
    Aₖ = kron(I(length(b)), Aₚ) + δt.*kron(A, Aₛ)

    # Runge-Kutta solves
    l = length(b)
    u_ =  Aₚ \ (Aₚ_ * u[k_])
    b_ = -As_*u[k_]
    k = Aₖ \ repeat(b_, l)
    s = [b[j]*k[(j-1)*Mn+1:j*Mn] for j in 1:l]
    append!(u, [u_ + δt.*sum(s)])
    Aₚ_ = Aₚ[:,:]
    As_ = Aₛ[:,:]
end

## 
# Fixed s comparisons (s=1/2, 1/3, 1/6)
##

us = []
Ap = [] # for plotting later
for t in [0., 0.5, 1.]
    # Construct approximation functions
    T̃ = ExtendedJacobi{T}(-frac_s(t),-frac_s(t))
    W = ExtendedWeightedJacobi{T}(frac_s(t),frac_s(t))
    L = AbsLaplacianPower(axes(T̃, 1), frac_s(t))
    P = L * T̃
    Q = L * W
    # Construct sum spaces
    Sₚ = SumSpace{T, Tuple{typeof(T̃), typeof(W)}}((T̃, W), intervals)
    S = SumSpace{T, Tuple{typeof(P), typeof(Q)}}((P, Q), intervals)

    # Construct least squares frame matrix
    @time Aₚ = Matrix(Sₚ[xc, 1:Mn]);
    append!(Ap, [Aₚ]) # Used for plotting later
    
    # Expand initial condition
    u₀ = Aₚ \ u0.(xc)

    # Construct least squares frame matrix
    @time Aₛ   = Matrix(S[xc, 1:Mn]);
    # Construct least squares Runge-Kutta frame matrix
    Aₖ = kron(I(length(b)), Aₚ) + δt.*kron(A, Aₛ)

    u = [u₀]
    # Run Runge-Kutta time loop
    @time for k_ = 1:λ
        l = length(b)
        b_ = -Aₛ*u[k_]
        k = Aₖ \repeat(b_, l)
        s = [b[j]*k[(j-1)*Mn+1:j*Mn] for j in 1:l]
        append!(u, [u[k_] + δt.*sum(s)])
    end
    append!(us, [u])
end


###
# Plotting
###
p = plot(xc, [Ap[1]*us[1][end] Ap[3]*u[end] Ap[2]*us[2][end] Ap[3]*us[3][end]],
        title="time = 1(s)", 
        label=[L"s=1/2" L"s=1/2-t/3" L"s=1/3" L"s=1/6"],
        legend=:topleft,
        xlabel=L"$x$",
        ylabel=L"$u(x)$",
        xtickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
        linewidth=2,
        markersize=3,
        linestyle=[:dash :solid  :dot :dashdot],
        gridlinewidth=2,
)
savefig("variable-s.pdf")