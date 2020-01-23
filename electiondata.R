electiondata <- read.csv(file.choose())
View(electiondata)
attach(electiondata)
# Linear regression technique can not be employed
prob1 <- predict(fit1,type="response")

logit <- glm(Result~Year+Amount.Spent+Popularity.Rank,data=electiondata,family = "binomial")
summary(logit)

exp(coef(logit))
prob <- predict(logit,type=c("response"),electiondata)
prob

confusion<-table(prob>0.5,electiondata$Result)
confusion

# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy
library(ROCR)
rocrpred<-prediction(prob,electiondata$Result)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(0,6))
# More area under the ROC Curve better is the logistic regression model obtained

