using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("ForwardDiff")
using CSV
using DataFrames
using ForwardDiff
using Random

df = CSV.read("..//dataset//Iris.csv", DataFrame)

#Declaring global varib
X = hcat(ones(150),Matrix(df[:,2:4])) #design matrix
θ = Vector{Int}(undef,size(df)[2])
rand!(θ,1:100)
y = df[:,5]

#hypothesis function
h(x) = θ'x

#cost function J
J(θ) = (0.5)(X*θ-y)'(X*θ-y)


function LMSUpdateRule(xi,yi,α,θj)
    θj .+= α*(yi - h(xi))*xi[j]
    end;

function BatchGradientDescent(X,y,α)
    
        for (idxθj,θj) in enumerate(θ)
            for (idx, xi) in enumerate(X)
                LMSUpdateRule(xi, y[idx], α, θj)
            end;
            θ[idxθj] = θj

        end;

    return θ
    end;




BatchGradientDescent(X,y,0.1)

println(J(θ))