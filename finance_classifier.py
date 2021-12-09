import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame

import random as rnd

from collections import Counter
from pandas.core.series import Series

import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_csv("../GiveMeSomeCredit/cs-training.csv")
test_df = pd.read_csv("../GiveMeSomeCredit/cs-test.csv")
# combine = [train_df, test_df]

# print(train_df.shape)
# print(test_df.shape)
# print(train_df.head())

# print(train_df.info())
# print(test_df.info())


# check null values
# print(pd.DataFrame({"Number of Null Values":train_df.isnull().sum(),
#              "Ratio":round(train_df.isnull().sum()/len(train_df)*100,2)}))

# print(pd.DataFrame({"Number of Null Values":test_df.isnull().sum(),
#              "Ratio":round(test_df.isnull().sum()/len(test_df)*100,2)}))

# Null values ratio: train: MonthlyIncome (19.82%) NumberOfDependents (2.62%) 
# Null values ratio: train: MonthlyIncome (19.81%) NumberOfDependents (2.59%)



# check duplicates: 
############################
# analysis findings:
# duplicated observations: 646
# observstions with same data but conflicting labels: 37
############################
# print(train_df.iloc[:,1:].columns)
# print(train_df.iloc[:, 1:].head(2))
# print(train_df.iloc[:, 1:].duplicated().value_counts())

train_df = train_df.iloc[:, 1:].drop_duplicates()
print('after dropping duplications', train_df.shape)
# print(train_df.iloc[:,1:].columns)
# print(train_df.iloc[:, 1:].head(2))
# print(train_df.iloc[:, 1:].duplicated().value_counts())

# check for outliers: an observation with 5 feature value < 1st quartile- (3rd quartile - 1st quartile) or > 3rd quartile + (3rd quartile - 1st quartile)
# are considered as outliers

def find_outlier(df,n,features):
    
    outlier_ind = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        
        IQR = Q3 - Q1
        
        # outlier range
        outlier_range = 1.5 * IQR      



        outlier_list_col = df[(df[col] < Q1 - outlier_range) | (df[col] > Q3 + outlier_range )].index    
        outlier_ind.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_ind)        
    real_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return real_outliers 


# drop outliers
outlier_indlst = find_outlier(train_df,5,train_df.columns.values[1:])
# print("outliers：",round(len(outlier_indlst)/train_df.shape[0]*100,2),"%")
# print(train_df.loc[outlier_indlst])
# print(train_df.shape)
# train_df = train_df.drop(outlier_indlst)
# print(train_df.shape)


# check prediction target
# SeriousDlqin2yrs
####################
# analysis findings: this is a imbalanced data set
# positive class: 6.684%
####################

# print('seriousDLqin2yrs ratio', train_df["SeriousDlqin2yrs"].sum()/len(train_df))
# plt.figure(figsize=(8,5))
# ax = sns.countplot(train_df['SeriousDlqin2yrs'])
# plt.xlabel('SeriousDlqin2yrs')
# plt.ylabel('Count')
# plt.show()



# analyze features

# age: 
#####################
#  analysis findings: yonger people tend to past due
# one borrower age = 0. This is an error in the data.
#  
#####################

# print(train_df.head(6))
# print(len(train_df.loc[train_df['age']<10]))
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df.loc[train_df['age'] > 0]["age"])
# # ax = sns.distplot(train_df.loc[train_df['SeriousDlqin2yrs'] == 1]['age'])
# ax = sns.distplot(train_df.loc[train_df['SeriousDlqin2yrs'] == 0]['age'])
# plt.show()


#NumberOfTime30-59DaysPastDueNotWorse
# NumberOfTimes90DaysLate
# NumberOfTime60-89DaysPastDueNotWorse
#####################
# analysis findings: borrowers with number of time >= 96 may be errors in the data and need to be confirmed
######################

# print(train_df["NumberOfTime30-59DaysPastDueNotWorse"].value_counts().sort_index())
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df['NumberOfTime30-59DaysPastDueNotWorse'])
# plt.xlabel('NumberOfTime30-59DaysPastDueNotWorse')
# plt.ylabel('Density')
# plt.show()

# print(train_df["NumberOfTime60-89DaysPastDueNotWorse"].value_counts().sort_index())
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df['NumberOfTime60-89DaysPastDueNotWorse'])
# plt.xlabel('NumberOfTime60-89DaysPastDueNotWorse')
# plt.ylabel('Density')
# plt.show()


# print(train_df["NumberOfTimes90DaysLate"].value_counts().sort_index())
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df['NumberOfTimes90DaysLate'])
# plt.xlabel('NumberOfTimes90DaysLate')
# plt.ylabel('Density')
# plt.show()



# DebtRatio

# print(train_df["DebtRatio"].quantile())
# print(np.percentile(train_df["DebtRatio"],25))
# print(np.percentile(train_df["DebtRatio"],75))
# print(train_df["DebtRatio"].quantile(0.8))


# MonthlyIncome
# No borrower's MonthlyIncome< 0 (invalid number)
# print(len(train_df.loc[train_df['MonthlyIncome'] < 0]))

# NumberOfOpenCreditLinesAndLoans
# print(train_df["NumberOfOpenCreditLinesAndLoans"].value_counts().sort_index())
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df['NumberOfOpenCreditLinesAndLoans'])
# plt.xlabel('NumberOfOpenCreditLinesAndLoans')
# plt.ylabel('Density')
# plt.show()


#  NumberRealEstateLoansOrLines
# print(train_df["NumberRealEstateLoansOrLines"].value_counts().sort_index())
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df['NumberRealEstateLoansOrLines'])
# plt.xlabel('NumberRealEstateLoansOrLines')
# plt.ylabel('Density')
# plt.show()


# NumberOfDependents
# print(train_df["NumberOfDependents"].value_counts().sort_index())
# plt.figure(figsize=(8,5))
# ax = sns.distplot(train_df['NumberOfDependents'])
# plt.xlabel('NumberOfDependents')
# plt.ylabel('Density')
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor


from sklearn.metrics import roc_auc_score, confusion_matrix, auc
from sklearn.metrics import roc_auc_score, precision_recall_curve,roc_curve

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

import six
import sys
sys.modules['sklearn.externals.six'] = six
from imblearn.over_sampling import SMOTE, ADASYN


data_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0)), ('scaler', StandardScaler())])



# def plot_roc_curve(fpr, tpr, label=None):
#     plt.figure(figsize=(8,6))
#     plt.plot(fpr,tpr,'b',label= 'AUC= %0.3f' % roc_auc) 
#     plt.legend(loc='lower right')
#     plt.plot([0,1],[0,1],'r--') # 
#     plt.axis([0,1,0,1])
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive rate")
#     plt.show()



x_train = train_df.iloc[:,1:]
y_train = train_df['SeriousDlqin2yrs'].astype('uint8')





data_train = data_pipeline.fit_transform(x_train)
x_train = data_train



# # check for outliers
# from sklearn.ensemble import IsolationForest
# clf = IsolationForest(random_state=0).fit_predict(x_train)
# print('outlier')
# print(Series(clf).value_counts())


# # create a parameter grid for ramdom sampling from
# from sklearn.model_selection import RandomizedSearchCV

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop =1000, num = 200)]
# # Number of features to consider at every split
# max_features = ['log2', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # print(random_grid)

# # rf = RandomForestClassifier()
# # rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 5, cv = 3, random_state=42, n_jobs = -1)


train_X, test_X, train_y, test_y = train_test_split(x_train,y_train,test_size=.1,random_state=42, stratify = y_train)

# # over sampling to address imbalanced data
# ros = SMOTE()
# x_train_res,y_train_res = ros.fit_resample(train_X, train_y)
# print('before resampling training data shape %s' % Counter(train_y))
# print('after resampling data shape %s' % Counter(y_train_res))

# # rfc = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rfc = RandomForestClassifier(n_estimators=441, min_samples_split=5, min_samples_leaf=2, max_features='log2', class_weight='balanced',
 max_depth=10, bootstrap=True, n_jobs=-1)


rfc.fit(train_X,train_y)
# # rf_random.fit(train_X, train_y)
# rfc.fit(x_train_res,y_train_res)

# # print('best parameters')
# # print(rf_random.best_params_)

# ###########
# #{'n_estimators': 441, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': 10, 'bootstrap': True}
# ###########



pred = rfc.predict_proba(test_X)[:,1]
# print('pred', pred[:10])
fpr, tpr, _ = roc_curve(test_y, pred)
roc_auc = auc(fpr,tpr)
# # plot_roc_curve(fpr,tpr)
print ('AUC Score :', roc_auc)


# make predictions on the test data
test_data_x = test_df.iloc[:,2:]
test_tansform_x = data_pipeline.fit_transform(test_data_x)

test_data_y = rfc.predict_proba(test_tansform_x)[:,1]
# print('test_df', test_df.shape)
# print('test_data_y', test_data_y.shape)
DataFrame(test_data_y).to_csv('result.csv', header=['Probability'])
# showresult = pd.read_csv('result.csv')
# print(showresult.head())


