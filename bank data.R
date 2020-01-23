bankdata <- read.csv(file.choose())
View(bankdata)
attach(bankdata)

logit <- glm(y~age+factor(default)+balance+factor(housing)+factor(loan)+day+duration+campaign+pdays+previous,data=bankdata,family = "binomial")        

summary(logit)
prob1 <- predict(fit1,type="response")
# Logistic Regression 
exp(coef(logit))


# Confusion matrix table 
prob <- predict(logit,type=c("response"),bankdata)
prob
confusion<-table(prob>0.5,bankdata$y)
confusion

# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy


# ROC Curve 
library(ROCR)
rocrpred<-prediction(prob,bankdata$y)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained
