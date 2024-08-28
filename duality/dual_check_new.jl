using JLD2, LinearAlgebra, CUDA, GilaElectromagnetics, BaryRational, Serialization
include("load_gila.jl")

(G::GlaOpr)(x::AbstractArray) = G * x
(G::AbstractMatrix)(x::AbstractArray) = G * x
adjoint(G::GlaOpr) = x -> conj(G * conj(x))
adjoint(G::Function) = x -> conj(G(conj(x)))

similar_fill(v::AbstractArray, fill_val::T) where T = fill!(similar(v), fill_val)
similar_fill(v::AbstractArray, dims::NTuple{N, Int}, fill_val::T) where {N, T} = fill!(similar(v, dims), fill_val)
Base.:\(::Nothing, x::AbstractArray) = (x, 0)


mutable struct ConstraintFunction
	size::Int
	𝕆::Function
	𝔼::Function
	𝕌::Function
	𝕌ᴴ::Function
	ℤᵀᵀ::Function
	ℤᵀˢ::Function
	S::AbstractVector{ComplexF64}
    ℝ¹::Diagonal{ComplexF32, CuArray{ComplexF32, 1, CUDA.DeviceMemory}}
    ℝ²::Diagonal{ComplexF32, CuArray{ComplexF32, 1, CUDA.DeviceMemory}}

	function ConstraintFunction(n::Int)
		Gᵒ = load_greens_operator((n, n, n), (1//32, 1//32, 1//32), set_type=ComplexF32, use_gpu=true)
		𝕍 = CUDA.Diagonal(CUDA.rand(ComplexF32, size(Gᵒ, 2))) - ComplexF32(0.5 * (1 + 1im)) * CUDA.Diagonal(CUDA.ones(ComplexF32, size(Gᵒ, 2)))
		ℝ¹ = CUDA.Diagonal(ComplexF32.(CUDA.rand(Float32, size(Gᵒ, 2)))) - ComplexF32(0.5) * CUDA.Diagonal(CUDA.ones(ComplexF32, size(Gᵒ, 2)))
		ℝ² = CUDA.Diagonal(ComplexF32.(CUDA.rand(Float32, size(Gᵒ, 2)))) - ComplexF32(0.5) * CUDA.Diagonal(CUDA.ones(ComplexF32, size(Gᵒ, 2)))
		S = CUDA.rand(ComplexF32, size(Gᵒ, 2))

		# Gᵒᴴ = Gᵒ'

		# 𝕍⁻¹ = inv(𝕍)
		# 𝕍⁻ᴴ = inv(𝕍)'
		𝕌 = x -> inv(𝕍)'(x) - Gᵒ'(x)
		𝕌ᴴ = x -> conj(𝕌(conj(x)))

		𝕆 = x -> ComplexF32(0.5) * (inv(𝕍)'(x) - inv(𝕍)(x))
		𝔼 = x::AbstractVector{ComplexF32} -> ComplexF32(0.5) * (𝕌(x) - 𝕌ᴴ(x))


		ℤᵀᵀ = x::AbstractVector{ComplexF32} -> 𝕆(x) + ComplexF32(0.5) * (𝕌(ℝ¹(x)) + conj(ℝ¹)𝕌ᴴ(x)) + ComplexF32(0.5) * (𝕌(ℝ²(x)) - conj(ℝ²)𝕌ᴴ(x))
		ℤᵀˢ = x -> ComplexF32(0.5) * ComplexF32.(ℝ¹(x) + im*ℝ²(x))
		new(size(Gᵒ, 2), 𝕆, 𝔼,  𝕌, 𝕌ᴴ, ℤᵀᵀ, ℤᵀˢ, S, ℝ¹, ℝ²)
	end
end


function eval(C::ConstraintFunction, ζ::Float32)
	ℤ = x -> C.ℤᵀᵀ(x) + ζ * C.𝔼(x)

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
			ŝ, precon_mvp = preconditioner \ residual
			mvp += precon_mvp
			# t = op * ŝ
			t = op(ŝ)
			mvp += 1
			ω = dot(t, s) / dot(t, t)
			ω = dot(t, residual) / dot(t, t)
			x += α*p̂ + ω*ŝ
			residual -= ω*t
			ρ_prev = ρ
			println(num_iter, " ", norm(residual))
		end
		# @show norm(residual), atol
		throw("BiCGStab did not converge after $max_iter iterations.")
	end
	T = bicgstab(ℤ, C.ℤᵀˢ(C.S), max_iter=C.size)[1]
    return real(imag(conj(C.S)'T) - conj(T)'C.𝔼(T))
end


function update(C::ConstraintFunction, dx::Float32)
    C.ℝ¹ += ComplexF32(dx) * CUDA.Diagonal(ComplexF32.(CUDA.rand(Float32, C.size))) - ComplexF32(0.5) * CUDA.Diagonal(CUDA.ones(ComplexF32, C.size))
    C.ℝ² += ComplexF32(dx) * CUDA.Diagonal(ComplexF32.(CUDA.rand(Float32, C.size))) - ComplexF32(0.5) * CUDA.Diagonal(CUDA.ones(ComplexF32, C.size))
    C.ℤᵀᵀ = x::AbstractVector{ComplexF32} -> C.𝕆(x) + ComplexF32(0.5) * (C.𝕌(C.ℝ¹(x)) + conj(C.ℝ¹)C.𝕌ᴴ(x)) + ComplexF32(0.5) * (C.𝕌(C.ℝ²(x)) - conj(C.ℝ²)C.𝕌ᴴ(x))
	C.ℤᵀˢ = x -> ComplexF32(0.5) * ComplexF32.(C.ℝ¹(x) + im*C.ℝ²(x))
end


function approximate(s::ConstraintFunction, z::Float32, width::Int, domain::AbstractVector{Float32}, codomain::AbstractVector{Float32}, tol=Float32(1e-3))
    domain_z = range(z - tol * abs(z), z + tol * abs(z), width)
    domain_new = Float32.(vcat(domain, domain_z))
    codomain_new = Float32.(vcat(codomain, map(x -> eval(s, x), domain_z)))
    return aaa(domain_new, codomain_new, clean=1), domain_new, codomain_new
end


function find_zero(s::ConstraintFunction, z::Float32, width_init::Int, width::Int, domain::AbstractVector{Float32}, codomain::AbstractVector{Float32}, tol=Float32(1e-3), search_depth=0)
    a, domain, codomain = approximate(s, z, width_init, domain, codomain, tol)
    _, _, zeros = prz(a)
    z = Float32(maximum(map(x -> real(x), zeros)))
    if abs(eval(s, z)) < tol || search_depth == 20
        return z, a, domain, codomain
    end
    # println("\tsearch_depth: ", search_depth)
    return find_zero(s, Float32(z), width, width, domain, codomain, tol, search_depth+1)
end


function optimize(s::ConstraintFunction, z::Float32, width_init::Int, width::Int, depth::Int, tol=Float32(1e-4), dx=Float32(1e-2))
    z, a, domain, codomain = find_zero(s, z, width_init, width, Array{Float32}(undef, 0), Array{Float32}(undef, 0), tol)
    errs = Array{Float32}(undef, depth, 3)
    for i in range(1, depth)
        println("depth: ", i)
        z, a, domain, codomain = find_zero(s, z, width, width, domain, codomain, tol)
        errs[i, :] = [z, abs(eval(s, z)), length(domain)]
        # update(s, dx)
    end
    return errs
end

C = ConstraintFunction(2)
optimize(C, Float32(1.0), 2, 2, 2, Float32(1e4))