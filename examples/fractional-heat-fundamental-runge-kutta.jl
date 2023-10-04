using FractionalFrames, LinearAlgebra
using Plots, LaTeXStrings

"""
Section 7.2

Solving the fractional heat equation
    (∂ₜ + (-Δ)ˢ) u = 0, u₀(x) = (1+x²)⁻ˢ.

We pick s=1/2 as we know the fundamental solution. We discretize in the time with
Runge-Kutta methods. We use
1. backward Euler (1st order)
2. Implicit midpoint rule (2nd order)
3. Gauss-Legendre(2) (4th order)
4. Gauss-Legendre(3) (6th order)
"""

# Different Butcher tableau
As = []; bs = [];

# Backward Euler
A = [1.]
b = [1.]
append!(As, [A]); append!(bs, [b])

# Implicit midpoint rule
A = [1/2]
b = [1.]
append!(As, [A]); append!(bs, [b])

# Gauss-Legendre(2)
A = [1/4 1/4-sqrt(3)/6;
    1/4+sqrt(3)/6 1/4]
b = [1/2 1/2]
append!(As, [A]); append!(bs, [b])

# Gauss-Legendre(3)
A = [5/36 2/9-sqrt(15)/15 5/36-sqrt(15)/30;
    5/36+sqrt(15)/24 2/9 5/36-sqrt(15)/24;
    5/36+sqrt(15)/30 2/9+sqrt(15)/15 5/36]
b = [5/18 4/9 5/18]
append!(As, [A]); append!(bs, [b])

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
M = 5001; Me = 5001; Mn = 211
xc = collocation_points(M, Me, I=intervals, endpoints=[-20*one(T),20*one(T)], innergap=1e-4)

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
# plot(xc, abs.(Aₚ*u₀ - u0.(xc)))

# λ is the inverse of the timestep δt
λs = [1e0, 1e1, 1e2, 1e3, 1e4];

# Store errors in matrix
errors = zeros(length(λs),4)

# Loop over Butcher tableau
for (A, b, j) in zip(As, bs, 1:length(bs))

    # Loop for decreasing time step length
    for (λ, i) in zip(λs, 1:length(λs))

        δt = inv(λ)

        # No need to evaluate lots of time-steps for the high-order methods
        if j == 3 && i == 5 || j == 4 && i ≥ 4
            print("Skipping method $j, timestep=$δt\n")
            errors[i,j] = NaN
            break
        end

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
        Y = zeros(length(xc), length(0:inv(λ):timesteps*inv(λ)))
        for j in 1:lastindex(u)
            t = inv(λ)*(j-1)
            Y[:,j] = y.(xc, t)
        end

        # Compute inf-norm relative error norm
        errors[i, j] = maximum(maximum(abs.(Aₚ*Us - Y), dims=1)'./maximum(abs.(Y), dims=1)')
        print("Method $j, timestep=$δt\n")
    end
end

plot(inv.(λs), errors,
    markershape=[:+ :diamond :dtriangle :circle],
    xlabel=L"$\delta t$",
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    ylabel=L"\mathrm{Error}",
    yscale=:log10,
    xscale=:log10,
    yaxis=[1e-11, 3e-1],
    label = ["Backward Euler" "Implicit midpoint" "Gauss-Legendre(2)" "Gauss-Legendre(3)"],
    legend =:topleft,
    xtick=[1e-4,1e-3,1e-2,1e-1,1e0],
    ytick=[1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
    palette=[:orangered, :skyblue2, :orange, :purple1]
)

# Compute and plot theoretical rates in dashed lines
rates = zeros(size(errors))
p = [1, 2, 4, 6]
for j in axes(rates,2)
    rates[:,j] = errors[1,j] .* (inv.(λs)).^p[j]
end
plot!(inv.(λs), rates,
    linestyle=:dash,
    # label = [L"$O(\delta t)$" L"$O(\delta t^2)$" L"$O(\delta t^4)$" L"$O(\delta t^6)$"],
    label = ["" "" "" ""],
    palette=[:orangered, :skyblue2, :orange, :purple1]
)


savefig("fractional-heat-convergence.pdf")
