using JLD2, LinearAlgebra, CUDA, GilaElectromagnetics
include("load_gila.jl")


# mutable struct ConstraintFunction
# 	function ConstraintFunction()
		
# 	end
# end


n = 512
Gᵒ = load_greens_operator((n, n, n), (1//32, 1//32, 1//32), set_type=ComplexF32, use_gpu=true)
# 𝕍 = CUDA.Diagonal(CUDA.rand(ComplexF32, size(Gᵒ, 2)))

𝕍 = CUDA.Diagonal(CUDA.rand(ComplexF32, size(Gᵒ, 2))) - ComplexF32(0.5 * (1 + 1im)) * CUDA.Diagonal(CUDA.ones(ComplexF32, size(Gᵒ, 2)))

#TODO make real
ℝ¹ = CUDA.Diagonal(CUDA.rand(ComplexF32, size(Gᵒ, 2))) - ComplexF32(0.5 * (1 + 1im)) * CUDA.Diagonal(CUDA.ones(ComplexF32, size(Gᵒ, 2)))
ℝ² = CUDA.Diagonal(CUDA.rand(ComplexF32, size(Gᵒ, 2))) - ComplexF32(0.5 * (1 + 1im)) * CUDA.Diagonal(CUDA.ones(ComplexF32, size(Gᵒ, 2)))
S = CUDA.rand(ComplexF32, size(Gᵒ, 2))


(G::GlaOpr)(x::AbstractArray) = G * x
(G::AbstractMatrix)(x::AbstractArray) = G * x
adjoint(G::GlaOpr) = x -> conj(G * conj(x))
adjoint(G::Function) = x -> conj(G(conj(x)))
# Gᵒᴴ = Gᵒ'

# 𝕍⁻¹ = inv(𝕍)
# 𝕍⁻ᴴ = 𝕍⁻¹'
𝕌 = x -> inv(𝕍)'(x) - Gᵒ'(x)
𝕌ᴴ = x -> conj(𝕌(conj(x)))

𝕆 = x -> ComplexF32(0.5) * (inv(𝕍)'(x) - inv(𝕍)(x))
𝔼 = x -> ComplexF32(0.5) * (𝕌(x) - 𝕌ᴴ(x))


ζ = ComplexF32(1.0)
ℤᵀᵀ = x -> 𝕆(x) + ComplexF32(0.5) * (𝕌(ℝ¹(x)) + conj(ℝ¹)𝕌ᴴ(x)) + ComplexF32(0.5) * (𝕌(ℝ²(x)) - conj(ℝ²)𝕌ᴴ(x))
ℤᵀˢ = x -> ComplexF32(0.5) * (ℝ¹(x) + im*ℝ²(x))
ℤ = x -> ℤᵀᵀ(x) + ζ * 𝔼(x)



# SOLVING THE SYSTEM
similar_fill(v::AbstractArray, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)
function bicgstab(op, b::AbstractVector; preconditioner=nothing, max_iter::Int=size(op, 2), atol::Real=zero(real(eltype(b))), rtol::Real=sqrt(eps(real(eltype(b)))))
	T = eltype(b)
	atol = max(atol, rtol * norm(b))

    mvp = 0 # mvp = matrix vector products

	x = similar_fill(b, zero(T))
	ρ_prev = zero(T)
	ω = zero(T)
	α = zero(T)
	v = similar_fill(b, zero(T))
	residual = deepcopy(b)
	residual_shadow = deepcopy(residual)
	p = deepcopy(residual)
	display(varinfo())
	s = similar(residual)

	for num_iter in 1:max_iter
		if norm(residual) < atol
			return x, mvp
		end

		ρ = dot(residual_shadow, residual)
		if num_iter > 1
			β = (ρ / ρ_prev) * (α / ω)
			p = residual + β*(p - ω*v)
		end
		p̂, precon_mvp = preconditioner \ p
        mvp += precon_mvp
		# v = op * p̂
		v = op(p̂)
        mvp += 1
		residual_v = dot(residual_shadow, v)
		α = ρ / residual_v
		residual -= α*v
		s = deepcopy(residual)

		if norm(residual) < atol
			x += α*p̂
			return x, mvp
		end

		ŝ, precon_mvp = preconditioner \ s
        mvp += precon_mvp
		# t = op * ŝ
        t = op(ŝ)
        mvp += 1
		ω = dot(t, s) / dot(t, t)
		x += α*p̂ + ω*ŝ
		residual -= ω*t
		ρ_prev = ρ
	end
    # @show norm(residual), atol
	throw("BiCGStab did not converge after $max_iter iterations.")
end

println(bicgstab(ℤ, ℤᵀˢ(S), max_iter=size(Gᵒ, 2)))