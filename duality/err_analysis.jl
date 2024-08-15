using Serialization, Plots

e = deserialize("errs.dat")
d = size(e)

hyperparameters = deserialize("hyperparameters.dat")
hyperparameters = reshape(map(identity, hyperparameters), d[1])


e_avg = reshape(maximum(e, dims = 2), d[1])
idx_good = findall(<(1e2), e_avg)
n = length(idx_good)
e_good = e[idx_good, :]
h_good = hyperparameters[idx_good, :]


s = reshape(map(h -> h[1], h_good), n)
w = reshape(map(h -> h[2], h_good), n)
e = reshape(maximum(e_good, dims = 2), n)
surface(s, w, e)
xlabel!("samples")
ylabel!("width")
savefig("preview.png")
println(maximum(e, dims = 1))
println(100 * size(e_good)[1] / d[1], " %")
println(sum(e_good, dims = 1))