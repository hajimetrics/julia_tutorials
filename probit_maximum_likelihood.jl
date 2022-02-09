using Plots
using StatsPlots
using Distributions
using Optim
using LinearAlgebra
using DataFrames
using CSV
using ForwardDiff

"""
Estimate Probit regression with Maximum Likelihood.

       y_name: dependent variable name
       X_name: independent variable name array
"""
function Probit_ML(data::DataFrame, y_name::String, X_name::Array)
       # Extract specified data
       y = data[:, y_name]
       X = Matrix(data[:, X_name])
       # add constant in X
       X = hcat(X, ones(size(X, 1)))

       ## Estimate Coefficients
       # initial β
       β_init = Vector{Float64}(undef, size(X, 2))
       # negative sum log likelihood
       # Strong Assumption: Normal(0, 1) for everyone (Homoscedasticity)
       # if you want Heteroscedasticity, rewrite the likelihood.
       function negsumloglike(β)
              nsll = -sum(log.(cdf.(Normal(0, 1), X * β)) .* (y) + log.(1 .- cdf.(Normal(0, 1), X * β)) .* (1 .- y))
              return nsll
       end
       # minimize nsll → get maximum likelihood estimator
       opt = optimize(negsumloglike, β_init)
       β_ml = Optim.minimizer(opt)

       ## Estimate Standard Error of Estimator
       # (based on Wooldridge p.481)
       β_vcm = inv(sum([(pdf(Normal(0, 1), transpose(X[i, :]) * β_ml)^2 * X[i, :] * transpose(X[i, :])) / (cdf(Normal(0, 1), transpose(X[i, :]) * β_ml) * (1 - cdf(Normal(0, 1), transpose(X[i, :]) * β_ml))) for i in 1:size(X, 1)]))
       β_se = sqrt.(Diagonal(β_vcm) * ones(size(β_ml)))

       z_values = β_ml ./ β_se
       ##
       result = DataFrame((var_name = push!(X_name, "(intercept)"), coef = β_ml, SE = β_se, z_value = z_values, p_value = (1 .- cdf.(Normal(0, 1), abs.(z_values))) .* 2))

       return result, β_vcm
end



# generating moc data
β_true = [0.6, -0.8, 1.5]
n = 1000
X = rand(Uniform(-1, 1), (n, 2))
X = hcat(ones(size(X, 1)), X)

# Data generating process 1
v_true = X * β_true + rand(Normal(0, 1), n) # rondomness here
y = v_true .>= 0

# Data generating process 2 (equivalent to process 1)
# p = cdf.(Normal(0, 1), X * β_true)
# y = rand.(Bernoulli.(p)) # rondomness here

# Estimate Model

β_hat, β_se_hat = Probit_ML(X[:, 2:3], y)

β_hat ./ β_se_hat

# Calculate Asymptotic Variance - Covariance
pdf(Normal(0, 1), transpose(X[1, :]) * β_hat)^2 * X[1, :] * transpose(X[1, :]) / (cdf(Normal(0, 1), transpose(X[1, :]) * β_hat) * (1 - cdf(Normal(0, 1), transpose(X[1, :]) * β_hat)))

Diagonal(sqrt(inv(sum([pdf(Normal(0, 1), transpose(X[i, :]) * β_hat)^2 * X[i, :] * transpose(X[i, :]) / (cdf(Normal(0, 1), transpose(X[i, :]) * β_hat) * (1 - cdf(Normal(0, 1), transpose(X[i, :]) * β_hat))) for i in 1:size(X, 1)]))))




# Read csv
# sample data is from https://stats.oarc.ucla.edu/r/dae/probit-regression/
# df = DataFrame(CSV.File("file.csv"))
df = CSV.File("binary.csv") |> DataFrame

# one hot encoding
df[:, "rank1"] = df[:, :rank] .== 1 .* 1.0
df[:, "rank2"] = df[:, :rank] .== 2 .* 1.0
df[:, "rank3"] = df[:, :rank] .== 3 .* 1.0
df[:, "rank4"] = df[:, :rank] .== 4 .* 1.0

df

y = df[:, :admit]
X_data = df[:, [:gre, :gpa, :rank2, :rank3, :rank4]]
X = Matrix(X_data)
X

res, β_vcm = Probit_ML(df, "admit", ["gre", "gpa", "rank2", "rank3", "rank4"])

print(res)
size(β_vcm)

(1 .- cdf.(Normal(0, 1), abs.(res[:, "z_value"]))) .* 2

res

# Marginal Effect
"""
Marginal Effect of probit regression for continuous variable

∂Pr(Y=1|X)/∂X_j = ∂Pr(Φ(Xβ))/∂X_j 
                = ϕ(Xβ)*β_j, where ϕ() is standard normal pdf

The marginal effect on X_j depends on β_j, X_j and other β and X. Therefore

"""
function average_marginal_effect_probit(result_df::DataFrame, x_margins::DataFrame, target_variable::String; is_discrete = false)
       X = Matrix(x_margins[:, result_df[1:(size(result_df, 1)-1), :var_name]])
       X = hcat(X, ones(size(X, 1)))
       β_hat = result_df[:, :coef]
       β_target = result_df[result_df[:, :var_name].==target_variable, :coef]

       if is_discrete
              # discrte process
              # discrete is f**king weird.
              # dydxs = nohting
       else
              dydxs = pdf.(Normal(0, 1), X * β_hat) .* β_target
       end
       ame = mean(dydxs)
       return ame
end



average_marginal_effect_probit(res, df, "gre") # 0.000447
average_marginal_effect_probit(res, df, "gpa") # 0.1552

# by R "margins"
#  factor     AME     SE       z      p   lower   upper
#     gpa  0.1552 0.0628  2.4723 0.0134  0.0322  0.2783
#     gre  0.0004 0.0002  2.1486 0.0317  0.0000  0.0009
#   rank2 -0.1564 0.0736 -2.1237 0.0337 -0.3007 -0.0121
#   rank3 -0.2868 0.0734 -3.9078 0.0001 -0.4306 -0.1429
#   rank4 -0.3213 0.0799 -4.0231 0.0001 -0.4778 -0.1647

# https://juliadiff.org/
# https://github.com/JuliaDiff/ForwardDiff.jl

X = Matrix(df[:, [:gre, :gpa, :rank2, :rank3, :rank4]])
function ame_func(β_hat::Vector)
       global X # call global X data outside
       X_t = hcat(X, ones(size(X, 1)))
       β_target = β_hat[1]
       dydxs = pdf.(Normal(0, 1), X_t * β_hat) .* β_target
       ame = mean(dydxs)
       return ame
end

# check ame function
ame_func(res[:, :coef])

#https://juliadiff.org/ForwardDiff.jl/stable/

ForwardDiff.gradient(ame_func, res[:, :coef])

g(res[:, :coef])