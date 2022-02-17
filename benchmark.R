# https://stats.oarc.ucla.edu/r/dae/probit-regression/
# https://cran.r-project.org/web/packages/margins/vignettes/Introduction.html

library(margins)

mydata <- read.csv("https://stats.idre.ucla.edu/stat/data/binary.csv")

## convert rank to a factor (categorical variable)
mydata$rank <- factor(mydata$rank)

## view first few rows
head(mydata)

summary(mydata)

# probit
myprobit <- glm(admit ~ gre + gpa + rank, family = binomial(link = "probit"), 
    data = mydata)

myprobit

## model summary
summary(myprobit)

# margins 
m <- margins(myprobit)
summary(m)
    
