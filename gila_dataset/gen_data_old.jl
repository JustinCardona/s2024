include("GilaOperators.jl")

using Base.Threads
using FFTW
using GLMakie
using GeometryBasics
using IterativeSolvers
using LinearAlgebra
using LinearAlgebra.BLAS
using Random
using Serialization
using Statistics
using ..GilaOperators
using DelimitedFiles
using GilaElectromagnetics

Random.seed!(0);

num_threads = nthreads()
BLAS.set_num_threads(num_threads)
FFTW.set_num_threads(num_threads)

Base.eltype(_::LippmanSchwinger) = ComplexF64
Base.size(op::LippmanSchwinger) = (prod(op.self_mem.trgVol.cel)*3, prod(op.self_mem.trgVol.cel)*3)
Base.size(op::LippmanSchwinger, _::Int) = prod(op.self_mem.trgVol.cel)*3

num_cells = 32#223
cells = [num_cells, num_cells, num_cells] # Cells in volume
scale = (1//100, 1//100, 1//100) # Size of cells in units of wavelength
coord = (0//1, 0//1, 0//1) # Center position of volume

"""
Solves t = (1 - XG)^{-1}i
"""
function solve(ls::LippmanSchwinger, i::AbstractArray{ComplexF64, 4}; solver=bicgstabl)
	out = solver(ls, reshape(deepcopy(i), prod(size(i))))
	return reshape(out, size(i))
end

# You can choose your chi as a function of space here
medium = fill(13.0 + 0.1im, num_cells, num_cells, num_cells, 1)
t1 = time()
ls = LippmanSchwinger(cells, scale, coord, medium)
lsa = LippmanSchwingerAdjoint(cells, scale, coord, medium)
elapsed_time = time() - t1
println("Lippman Schwinger: ", elapsed_time, " seconds")


mutable struct Us <: GilaOperator
	ls::LippmanSchwinger
	lsa::LippmanSchwinger
end

function Base.:*(op::Us, x::AbstractVector{ComplexF64})
	x_copy = deepcopy(x)
	return vec(0.5 * (vec(source).*(op.ls*deepcopy(x)) + vec(op.lsa*source).*x_copy))
end

function Base.:*(op::Us, x::AbstractArray{ComplexF64, 4})
	x_copy = deepcopy(x)
	return 0.5 * (source.*(op.ls*x_copy) + (op.lsa*source).*x_copy)
end

Base.eltype(_::Us) = ComplexF64
Base.size(op::Us) = (prod(op.ls.self_mem.trgVol.cel)*3, prod(op.ls.self_mem.trgVol.cel)*3)
Base.size(op::Us, _::Int) = prod(op.ls.self_mem.trgVol.cel)*3

source = zeros(ComplexF64, num_cells, num_cells, num_cells, 3)
i = 14
while i >= 0
	global source = 1e-2 * rand(ComplexF64, num_cells, num_cells, num_cells, 3)
	writedlm( "data/p_"*string(i)*".csv",  source, ',')	
	function U_s(x)		
		return vec(0.5 * (source.*(ls*x) + (lsa*source).*x))
	end
	eigval = powm(Us(ls, lsa), inverse=true, tol = 1e-4)
	writedlm( "data/e_"*string(i)*".csv",  eigval, ',')
	global i += 1
end

