# Titanic - Machine Learning from disaster

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 15:30:28 2018

@author: Jon Goni
"""

# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_absolute_error

# Read dataframes
data = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

# Select predictors
predictors = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
data = data.dropna(subset=[predictors])
X_train = data[predictors]
y_train = data.Survived

# Model definition
model = LogisticRegression()

# Fit model to the data
model.fit(X_train, y_train)

# Apply model to first m examples
m = 10
print('Making predictions for these {} passengers:'.format(m))
print(X_train.head(m))
print('The predicted values are:')
print(model.predict(X_train.head(m)))

# Create submission
y_test = model.predict(X_test[predictors])
print(y_test)

# Prepare submission file
my_submission = pd.DataFrame({'Id': X_test.PassengerId,'Survived': y_test})
my_submission.to_csv('submission.csv', index=False)