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

Random.seed!(0);

num_threads = nthreads()
BLAS.set_num_threads(num_threads)
FFTW.set_num_threads(num_threads)

num_cells = 32
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
medium = fill(5.0 + 0im, num_cells, num_cells, num_cells, 1)
ls = LippmanSchwinger(cells, scale, coord, medium)

## VISUALIZATION
source = zeros(ComplexF64, num_cells, num_cells, num_cells, 3)

# Sources
A = 10.0
for i in 5:25
	source[5, 7, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
	source[6, 7, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
	source[5, 8, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
	source[6, 8, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]

	source[10, 7, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
	source[11, 7, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
	source[10, 8, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
	source[11, 8, i, :] = A * [0.0+0.0im, 0.0+0.0im, 1.0+0.0im]
end

points = [Point3f((x-1)*scale[1] + coord[1], (y-1)*scale[2] + coord[2], (z-1)*scale[3] + coord[3]) for x in 1:num_cells for y in 1:num_cells for z in 1:num_cells]
field = [Vec3f(real.(source[x, y, z, :])...) for x in 1:num_cells for y in 1:num_cells for z in 1:num_cells]

out = solve(ls, source)
field_out = [Vec3f(real.(out[x, y, z, :])...) for x in 1:num_cells for y in 1:num_cells for z in 1:num_cells]

which_field = field_out

view_scale = 1000
grid_points = [Point3f(view_scale*((x-1)*scale[1] + coord[1]), view_scale*((y-1)*scale[2] + coord[2]), view_scale*((z-1)*scale[3] + coord[3])) for x in 1:num_cells for y in 1:num_cells for z in 1:num_cells]
cube = Rect3f((0.0, 0.0, 0.0), scale .* view_scale)

color = norm.(which_field)
color[color .< 1e-1] .= NaN

scene = arrows(grid_points, which_field; color=color, arrowsize=(view_scale * 0.3 * minimum(scale), view_scale * 0.3 * minimum(scale), view_scale  * 0.5 * minimum(scale)))
display(scene)
readline()