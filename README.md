# nsbk2021
# repository: credit-predict


# Training and Predication:

Run finance_classifier.py 

Note:  
1. This will (a) do training on "cs-training.csv", (b) make prediction on "cs-test.csv", (c) output prediction results to "result.csv"
2. Pls. make sure "cs-training.csv" and "cs-test.csv" are in directory: "../GiveMeSomeCredit/".
3. Python version: 3.6.10.   Sklearn version: 0.23.1

# Data Analysis:

1. Check for NULL calues. (Null values ratio: train: MonthlyIncome (19.82%) NumberOfDependents (2.62%) ) NULL values are filled by various strateties such as mean and zero. Zero has a better performance.
2. Check duplicates. There are duplications if the first column (the index column) is excluded. There are 609 duplications. Duplications are dropped from the data. In addition, 37 rows have conflicting labels, i.e., two observations (borrowers) have the same value for all the ten features but with different labels ('SeriousDlqin2yrs').
3. Check for outliers. An observation with more than 5 outlier featue values are considered as an outlier. When outliers are dropped, the AUC value is a little bit lower.
4. Check if the data is balanced. The data is not balanced.The positive class (SeriousDlqin2yrs=1) is 6.684% of the total data.
5. Check the distributions of all the ten features. One observation has age 0, which is an error. Yonger age borrowers tend to pass due. For 'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse', observations with number of time >= 96 may be errors in the data and need to be confirmed.

# Build classifier:

1. Use random forest as the classifier.
2. Use grid search to find the best parameters for the classifier.
3. Use oversampling and assign class weights when training the classifier to deal with imbalanced data.

# Evaluation:

1. Perform 5-fold cross validation on the training data.
2. Evaluated using AUC score. Average score of 5 folds is 0.859.

# Prediction:
Predictions to data in cs-test.csv are stored in result.csv.

# Cross validation:

5-fold cross validation using only training data is available in: finance_classifier_5foldcv.py.
