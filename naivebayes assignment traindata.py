# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 10:50:34 2022

@author: vaishnav
"""
#================================================================================================================
#importing the data

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:\\anaconda\\New folder (2)\\SalaryData_Train.csv")
df

list(df)

df.info()
df.describe()

df.dtypes
#=====================================================================================================================

#Finding the special characters in the data frame 

df.isin(['?']).sum(axis=0)
print(df[0:5])


df.native.value_counts()
df.native.unique()


df.workclass.value_counts()
df.workclass.unique()


df.occupation.value_counts()
df.occupation.unique()


df.sex.value_counts()

#==================================================================================================================
#visualisation
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(df)

t1 = pd.crosstab(index=df["education"],columns=df["workclass"])
t1.plot(kind='bar')

t2 = pd.crosstab(index=df["education"],columns=df["Salary"])
t2.plot(kind='bar')


t3 = pd.crosstab(index=df["sex"],columns=df["race"])
t3.plot(kind='bar')


t4 = pd.crosstab(index=df["maritalstatus"],columns=df["sex"])
t4.plot(kind='bar')

df["age"].hist()
df["educationno"].hist()
df["capitalgain"].hist()
df["capitalloss"].hist()
df["hoursperweek"].hist()


# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))
# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

#========================================================================================================================================
#splitting the data into x and y

X = df.drop(['Salary'], axis=1)

y = df['Salary']

#======================================================================================================================

# split X and y into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#==============================================================================================================================================

# display categorical variables

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

# display numerical variables

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

#==============================================================================================================================================

# print categorical variables with missing data

for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, (X_train[col].isnull().mean()))
# impute missing categorical variables with most frequent value

for df2 in [X_train, X_test]:
    df2['workclass'].fillna(X_train['workclass'].mode()[0], inplace=True)
    df2['occupation'].fillna(X_train['occupation'].mode()[0], inplace=True)
    df2['native'].fillna(X_train['native'].mode()[0], inplace=True)  
    
    
#==============================================================================================================================================    
    
import category_encoders as ce
# encode variables with one-hot encoding

encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'maritalstatus', 'occupation', 'relationship', 
                                 'race', 'sex', 'native'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

#==============================================================================================================================================

cols = X_train.columns
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])
X_train.head()

#==============================================================================================================================================

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

# fit the model
gnb.fit(X_train, y_train)
GaussianNB()
y_pred_test = gnb.predict(X_test)

from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#==============================================================================================================================================

from sklearn.naive_bayes import BernoulliNB
bn = BernoulliNB()

bn.fit(X_train, y_train)

y_pred_test = bn.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#==============================================================================================================================================

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)

#prediction
y_pred_test = logreg.predict(X_test)

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#==============================================================================================================================================






