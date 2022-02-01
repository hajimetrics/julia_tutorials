using Plots
using StatsPlots
using Distributions
using Optim


"""
Estimate Probit regression with Maximum Likelihood.

       X: independent variable matrix
       y: dependent variable vector
"""
function Probit_ML(X, y)
       
       # add constant in X
       X = hcat(ones(size(X, 1)), X)

       ## Estimate Coefficients
       # initial β
       β_init = Vector{Float64}(undef, size(X, 2))
       # negative sum log likelihood
       function negsumloglike(β)
              nsll = -sum(log.(cdf.(Normal(0, 1), X * β)) .* (y) + log.(1 .- cdf.(Normal(0, 1), X * β)) .* (1 .- y))
              return nsll
       end
       # minimize nsll → get maximum likelihood estimator
       opt = optimize(negsumloglike, β_init)
       β_ml = Optim.minimizer(opt)

       ## Estimate Standard Error of Estimator
       # (based on Wooldridge p.481)
       β_vcm = inv(sum([pdf(Normal(0,1),transpose(X[i,:])*β_hat)^2 * X[i,:] * transpose(X[i,:])/ (cdf(Normal(0,1),transpose(X[i,:])*β_hat)*(1 - cdf(Normal(0,1),transpose(X[i,:])*β_hat))) for i in 1:size(X,1)]))

       return β_ml
end


plot(Normal(10, 7))
dist = Normal(0,1)
pdf(dist,0.5)
X = [1 3 ;4 6 ;7 9,]
hcat(ones(size(X, 1)), X)
y = [0, 1, 0]

v = X * Vector{Float64}(undef, size(X, 2))
sum(pdf.(Normal(0,1), v).^(y) + (1 .- pdf.(Normal(0,1), v)).^(1 .- y))
pdf(Normal(0,1),v[1])*(y[1]) +  (1 - pdf(Normal(0,1),v[1]))*(1 .- y[1]) 
pdf.(Normal(0, 1), v) .* (y) + (1 .- pdf.(Normal(0, 1), v)) .* (1 .- y)



function negsumloglike(β)
       nsll = -sum(log.(cdf.(Normal(0, 1), X * β)) .* (y) + log.(1 .- cdf.(Normal(0, 1), X * β)) .* (1 .- y))
       return nsll
end


#function negsumfor(β)
       nsll = 0
       for i in 1:size(X, 1)
              nsll += pdf(Normal(0, 1), )
       end
       nsll = -sum(log.(pdf.(Normal(0, 1), X * β)) .* (y) + log.(1 .- pdf.(Normal(0, 1), X * β)) .* (1 .- y))
       return nsll
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

β_hat  = Probit_ML(X[:,2:3],y)

# Calculate Asymptotic Variance - Covariance
pdf(Normal(0,1),transpose(X[1,:])*β_hat)^2 * X[1,:] * transpose(X[1,:])/ (cdf(Normal(0,1),transpose(X[1,:])*β_hat)*(1 - cdf(Normal(0,1),transpose(X[1,:])*β_hat)))

Diagonal(sqrt(inv(sum([pdf(Normal(0,1),transpose(X[i,:])*β_hat)^2 * X[i,:] * transpose(X[i,:])/ (cdf(Normal(0,1),transpose(X[i,:])*β_hat)*(1 - cdf(Normal(0,1),transpose(X[i,:])*β_hat))) for i in 1:size(X,1)]))))



X_obs = X[:, 2:3]
Probit_ML(X_obs,y)

β_init = Vector{Float64}(undef, size(X, 2))

v = X * β_true
pdf(Normal(0, 1), v[1])
pdf(Normal(0, 1), v[2])
pdf.(Normal(0, 1), v)

log.(pdf.(Normal(0, 1), v)) .* (y) 


opt = optimize(negsumloglike, β_init)
Optim.minimizer(opt)

f(x) = (x[1] - 2.0) ^ 2
opt = optimize(f, [0.0])
Optim.minimizer(opt)

print(Probit_ML(X,y))



# 