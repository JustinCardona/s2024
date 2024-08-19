using Serialization, Plots



e = deserialize("errs_1000.dat")
errs = e[:, :, 3]
println(size(errs, 1))


function diff(x)
    y = vcat([0], x)
    return map(j -> y[j] - y[j-1], range(2, length(y)))   
end
samples = map(i -> diff(errs[i, :]), range(1, size(errs, 1)))
samples = foldl(hcat, samples)'

histogram(samples[:, 1])
savefig("preview.png")