using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
using CSV
using DataFrames

df = CSV.read("../dataset/Iris.csv", DataFrame)

#X = df[:,2:4] #design matrix 
X = hcat(ones(150),Matrix(df[:,2:4])) #design matrix

var = [[1,2,3] [4,5,6]]
println(var)

#with open(""../Dataset/Iris.csv")

#hypothesis function
h(x,θ) = θ'x

#cost function J
J(θ) = (0.5)(Xθ-y)'(Xθ-y)

#\

function BatchDescendingFunc(x)


    end;

function BatchGradientDescent(X,y,iteration,learning_rate)
    α = learning_rate
    θ = ones(size(X, 2))
    
    for epoch in epochs
        θj = X[:]
        idxθj = 
        #result = map(θ -> 
    


