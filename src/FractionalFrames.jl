module FractionalFrames

using ClassicalOrthogonalPolynomials, ContinuumArrays, HarmonicOrthogonalPolynomials, HypergeometricFunctions, MultivariateOrthogonalPolynomials,
    SpecialFunctions, StaticArrays, LinearAlgebra

import Base: in, axes, getindex, ==, oneto, *, \, +, -, convert, broadcasted
import ClassicalOrthogonalPolynomials: ∞, Derivative, jacobimatrix, @simplify, HalfLine, Weight, orthogonalityweight, recurrencecoefficients, ℝ, OneToInf
import ContinuumArrays: Basis, QuasiAdjoint, AbstractQuasiArray, ApplyQuasiMatrix
import HarmonicOrthogonalPolynomials: AbsLaplacianPower
import HypergeometricFunctions: _₂F₁general2

include("extendedchebyshev.jl")
include("extendedjacobi.jl")
include("sumspace.jl")
include("symmetriclaguerre.jl")
include("extendedhermite.jl")
include("extendedzernike.jl")

export  ∞, oneto, Block, Derivative, Weighted, AbsLaplacianPower,
        ExtendedChebyshev, ExtendedChebyshevT, ExtendedChebyshevU, extendedchebyshevt, ExtendedWeightedChebyshevT, ExtendedWeightedChebyshevU,
        jacobimatrix,
        ExtendedJacobi, ExtendedWeightedJacobi, DerivativeExtendedJacobi, DerivativeExtendedWeightedJacobi, GeneralExtendedJacobi, 
        ExtendedHermite, ExtendedNormalizedHermite,
        SymmetricLaguerre, ExtendedSymmetricLaguerre, SymmetricLaguerreWeight,
        ExtendedZernike, ExtendedWeightedZernike,
        collocation_points, Interlace,
        SumSpace

# Affine transform to scale and shift polys. 
affinetransform(a,b,x) = 2 /(b-a) * (x-(a+b)/2)

at(a,b,x) = (b-a)/2 * x .+ (b+a)/2

# Construct collocation points
function collocation_points(M::Int, Me::Int; I::AbstractVector=[-1.,1.], endpoints::AbstractVector=[-5.,5.], innergap::Real = 0., remove_endpoints::Bool=false)
    Tp = eltype(I)
    el_no = length(I)-1

    x = Array{Tp}(undef,el_no*M+2*Me)
    # xnodes = LinRange{Tp}(innergap,1-innergap,M)
    # chebnodes = sort(cos.(π.*xnodes))

    xxnodes = LinRange{Tp}(-1+innergap,1-innergap,M)
    for el = 1:el_no
        x[(el-1)*M+1:el*M] = at(I[el], I[el+1], xxnodes) 
    end
    # xnodes = LinRange{Tp}(innergap,1-innergap,Me)
    # chebnodes = sort(cos.(π.*xnodes))

    xxnodes = LinRange{Tp}(-1+innergap,1-innergap,Me)
    x[el_no*M+1:el_no*M+Me] = at(endpoints[1], I[1], xxnodes) 
    x[el_no*M+1+Me:el_no*M+2*Me] = at(I[end],endpoints[2],xxnodes)

    if remove_endpoints
        filter!(x->x∉I, x)
    end
    return sort(unique(x))
end

end # module FractionalFrames