library(ggplot2)
library(mlbench)
library(reticulate)

#use_virtualenv("myenv/")
kerneloptimizer <- import("kerneloptimizer")
optimizer <- kerneloptimizer$optimizer$KernelOptimizer

p <- mlbench.spirals(500,1,sd=.03)
#p <- mlbench.2dnormals(500) # Base com superposição
X <- p$x
y <- as.numeric(p$classes)
d <- ncol(X)

# L is needed after the number, as we want to pass integers to Python
print("Initializing MLP Kernel")
opt <- optimizer(kernel='mlp',input_dim=d,hidden_dim=20L,output_dim=50L)
print("Training MLP")
opt$fit(X,y,n_epochs=500L)
lspace <- opt$get_likelihood_space(X,y)

# ggplot(lspace, aes(x=sim_to_c1.0,y=sim_to_c2.0,color=as.factor(y))) + geom_point()

print("Initializing Gaussian Kernel")
opt2 <- optimizer(kernel='gaussian')
print("Finding the width")
opt2$fit(X,y)
print(opt2$width)

lspace2 <- opt2$get_likelihood_space(X,y)

# ggplot(lspace2, aes(x=sim_to_c1.0,y=sim_to_c2.0,color=as.factor(y))) + geom_point()
