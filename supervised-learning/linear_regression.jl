using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
using CSV
using DataFrames

df = CSV.read("../dataset/Iris.csv", DataFrame)

X = df[:,2:4]

var = [[1,2,3] [4,5,6]]
println(var)

#with open(""../Dataset/Iris.csv")

h(x,θ) = θ

println(X)

