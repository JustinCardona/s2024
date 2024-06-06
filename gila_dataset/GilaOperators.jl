module GilaOperators

using LinearAlgebra
using Serialization
using Statistics
using GilaElectromagnetics

export GilaOperator, LippmanSchwinger, LippmanSchwingerAdjoint, YGi

abstract type GilaOperator end

function LinearAlgebra.mul!(y::AbstractVector{ComplexF64}, op::GilaOperator, x::AbstractVector{ComplexF64})
	y .= op * x
	return y
end

mutable struct LippmanSchwinger <: GilaOperator
	self_mem::GlaOprMem
	medium::AbstractArray{ComplexF64, 4}
end

internal_size(op::LippmanSchwinger) = (op.self_mem.trgVol.cel..., 3)

function Base.:*(op::LippmanSchwinger, x::AbstractVector{ComplexF64})
	x_copy = deepcopy(x)
	acted_vec = egoOpr!(op.self_mem, reshape(deepcopy(x), internal_size(op)))
	acted_vec .*= op.medium
	acted_vec .= reshape(x_copy, size(acted_vec)) .- acted_vec
	return reshape(acted_vec, size(x))
end

function Base.:*(op::LippmanSchwinger, x::AbstractArray{ComplexF64, 4})
	x_copy = deepcopy(x)
	acted_vec = egoOpr!(op.self_mem, deepcopy(x))
	acted_vec .*= op.medium
	acted_vec .= x_copy .- acted_vec
	return acted_vec
	# return reshape(acted_vec, prod(size(x)))
end

Base.eltype(_::LippmanSchwinger) = ComplexF64
Base.size(op::LippmanSchwinger) = (prod(op.self_mem.trgVol.cel)*3, prod(op.self_mem.trgVol.cel)*3)
Base.size(op::LippmanSchwinger, _::Int) = prod(op.self_mem.trgVol.cel)*3


function LippmanSchwingerAdjoint(cells::AbstractVector{Int}, scale::NTuple{3, Rational{Int}}, coord::NTuple{3, Rational{Int}}, medium::AbstractArray{ComplexF64, 4})
	options = GlaKerOpt(1.0 + 0.0im, 32, true, false, (), ())
	self_volume = GlaVol(cells, scale, coord)
	filename = "preload/$(cells[1])x$(cells[2])x$(cells[3])_$(float(scale[1]))x$(float(scale[2]))x$(float(scale[3]))@$(float(coord[1])),$(float(coord[2])),$(float(coord[3])).fur"
	if isfile(filename)
		fourier = deserialize(filename)
		self_mem = GlaOprMem(options, self_volume, egoFur=fourier, setType=ComplexF64)
	else
		self_mem = GlaOprMem(options, self_volume, setType=ComplexF64)
		serialize(filename, self_mem.egoFur)
	end
	return LippmanSchwinger(self_mem, medium)
end


function LippmanSchwinger(cells::AbstractVector{Int}, scale::NTuple{3, Rational{Int}}, coord::NTuple{3, Rational{Int}}, medium::AbstractArray{ComplexF64, 4})
	options = GlaKerOpt(false)
	self_volume = GlaVol(cells, scale, coord)
	filename = "preload/$(cells[1])x$(cells[2])x$(cells[3])_$(float(scale[1]))x$(float(scale[2]))x$(float(scale[3]))@$(float(coord[1])),$(float(coord[2])),$(float(coord[3])).fur"
	if isfile(filename)
		fourier = deserialize(filename)
		self_mem = GlaOprMem(options, self_volume, egoFur=fourier, setType=ComplexF64)
	else
		self_mem = GlaOprMem(options, self_volume, setType=ComplexF64)
		serialize(filename, self_mem.egoFur)
	end
	return LippmanSchwinger(self_mem, medium)
end
end


