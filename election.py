import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
#Importing Data
election = pd.read_csv("file:///E:/excelr/Excelr Data/Assignments/Logisitc Regression/election_data.csv")
print(election)
election.head()
#removing CASENUM
election.drop(["Election-id"],axis=1) 
temp = election.drop(["Election-id"],axis=1,inplace=True)
?election.drop

##temp = claimants.drop(claimants.index[[0,1,2,3]],axis=0)
election.head(4)# to see top 10 observations

# usage lambda and apply function
# apply function => we use to apply custom function operation on 
# each column
# lambda just an another syntax to apply a function on each value 
# without using for loop 
election.isnull().sum()
sns.boxplot(x="Result",y="Year",data=election)

plt.boxplot(election.Year)

plt.boxplot(election.AS)
plt.boxplot(election.PR)

election.describe()

##sample to apply
election.apply(lambda x:x.mean()) 
election.mean()

#Imputating the missing values with most repeated values in that column  

# lambda x:x.fillna(x.value_counts().index[0]) 
# the a





bove line gives you the most repeated value in each column  

election.Result.value_counts()
election.Result.value_counts().index[0]#1 # gets you the most occuring value
election.Result.mode()[0]#1
election.isnull().sum()
election.isna().sum() 
election.iloc[:,0:0] = Result.iloc[:,0:0].apply(lambda x:x.fillna(x.mode()[0]))
election.isnull().sum()
#claimants.SEATBELT = claimants.SEATBELT.fillna(claimants.SEATBELT.value_counts().index[0])

election.iloc[:,0:0].columns

election.Year = election.Year.fillna(election.Year.mean())

# filling the missing value with mean of that column
election.iloc[:,1:] = election.iloc[:,1:].apply(lambda x:x.fillna(x.mean()))

election.AS=election.AS.fillna(election.AS.mean())
election.iloc[:,2:] = election.iloc[:,2:].apply(lambda x:x.fillna(x.mean()))

election.PR=election.PR.fillna(election.PR.mean())
election.iloc[:,3:] = election.iloc[:,3:].apply(lambda x:x.fillna(x.mean()))

# Checking if we have na values or not 
election.isnull().sum() # No null values
from scipy import stats
import scipy.stats as st
st.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


#Model building 

import statsmodels.formula.api as sm
logit_model=sm.logit('Result~Year+AS+PR',data=election).fit()


#summary
logit_model.summary()
y_pred = logit_model.predict(election)

election["pred_prob"] = y_pred
# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
election["Att_val"] = 0

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
election.loc[y_pred>=0.5,"Att_val"] = 1

election.Att_val

from sklearn.metrics import classification_report
classification_report(election.Att_val,election.Result)

# confusion matrix 
confusion_matrix = pd.crosstab(election['Result'],election.Att_val)

confusion_matrix
accuracy = (4+3)/(9) # 0.7777
accuracy

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(election.Result, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 


### Dividing data into train and test data sets
election.drop("Att_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train,test = train_test_split(election,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('Result~Year+AS+PR',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train.iloc[:,1:])

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
train["train_pred"] = np.zeros(938)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1


# confusion matrix 
confusion_matrix = pd.election(train['Result'],train.train_pred)

confusion_matrix
accuracy_train = (3+4)/(9) # 0.77777777
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)

# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
test["test_pred"] = np.zeros(402)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['Result'],test.test_pred)

confusion_matrix
accuracy_test = (7)/(9) # 0.7777777
accuracy_test

