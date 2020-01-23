affairs <- read.csv(file.choose())
View(affairs)
attach(affairs)
logit <- glm(affair~factor(gender)+age+yearsmarried+factor(children)+religiousness+education+occupation+rating,data=affairs,family="binomial")
summary(logit)
logit1 <- glm(affair~age+yearsmarried+religiousness+rating+factor(children),data=affairs,family="binomial")
summary(logit1)
logit2 <- glm(affair~yearsmarried+religiousness+rating+education+occupation,data=affairs,family="binomial")
summary(logit2)
logit3 <- glm(affair~factor(gender)+age+yearsmarried+religiousness+education+occupation+rating,data=affairs,family="binomial")
summary(logit3)
logit4 <- glm(affair~factor(gender)+age+yearsmarried+factor(children)+religiousness+occupation+rating,data=affairs,family="binomial")
summary(logit4)
logit5 <- glm(affair~factor(gender)+age+yearsmarried+factor(children)+religiousness+education+rating,data=affairs,family="binomial")
summary(logit5)
logit6 <- glm(affair~factor(gender)+age+yearsmarried+factor(children)+religiousness+rating,data=affairs,family="binomial")
summary(logit6)


# Odds Ratio
exp(coef(logit6))


# Confusion matrix table 
prob <- predict(logit6,type=c("response"),affairs)
prob
confusion<-table(prob>0.5,affairs$affair)
confusion

# Model Accuracy 
Accuracy<-sum(diag(confusion)/sum(confusion))
Accuracy


# ROC Curve 
library(ROCR)
rocrpred<-prediction(prob,affairs$affair)
rocrperf<-performance(rocrpred,'tpr','fpr')
plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained
