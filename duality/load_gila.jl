function get_preload_dir(dir="preload")
	found_dir = false
	for i in 1:10
		if !isdir(dir)
			dir = "../"^i * "preload/"
		else
			found_dir = true
			break
		end
	end
	if !found_dir
		error("Could not find preload directory. Please create a directory named 'preload' in the current directory or parent directories.")
	end
	return dir
end

function load_greens_operator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}; set_type=ComplexF64, use_gpu::Bool=false)
	preload_dir = get_preload_dir()
	type_str = set_type == ComplexF64 ? "c64" : (set_type == ComplexF32 ? "c32" : "c16")
	fname = "$(type_str)_$(cells[1])x$(cells[2])x$(cells[3])_$(scale[1].num)ss$(scale[1].den)x$(scale[2].num)ss$(scale[2].den)x$(scale[3].num)ss$(scale[3].den).jld2"
	fpath = joinpath(preload_dir, fname)
	if isfile(fpath)
		file = jldopen(fpath)
		fourier = file["fourier"]
		if use_gpu
			fourier = CuArray.(fourier)
		end
		options = GlaKerOpt(use_gpu)
		volume = GlaVol(cells, scale, (0//1, 0//1, 0//1))
		mem = GlaOprMem(options, volume; egoFur=fourier, setTyp=set_type)
		return GlaOpr(mem)
	end
	operator = GlaOpr(cells, scale; setTyp=set_type, useGpu=use_gpu)
	fourier = operator.mem.egoFur
	if use_gpu
		fourier = Array.(fourier)
	end
	jldsave(fpath; fourier=fourier)
	return operator
end
