electinon <- read.csv(file.choose())
View(electinon)

attach(electinon)
library(caTools)
split <- sample.split(electinon, splitRatio = 0.8)