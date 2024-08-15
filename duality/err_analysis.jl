using Serialization, Plots

e = deserialize("errs.dat")
d = size(e)

hyperparameters = deserialize("hyperparameters.dat")
hyperparameters = reshape(map(identity, hyperparameters), d[1])


e_avg = reshape(sum(e, dims = 2), d[1]) / d[2]
idx_good = findall(<(1e1), e_avg)
n = length(idx_good)
e_good = e[idx_good, :]
h_good = hyperparameters[idx_good, :]


s = reshape(map(h -> h[1], h_good), n)
w = reshape(map(h -> h[2], h_good), n)
e = reshape(sum(e_good, dims = 2) / d[2], n)
println(d)
surface(s, w, e)
xlabel!("samples")
ylabel!("width")
# histogram(e)
savefig("preview.png")
println(100 * n / d[1], " %")
println(sum(e_good, dims = 1) / n)