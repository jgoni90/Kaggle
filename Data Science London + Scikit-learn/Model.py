# -*- coding: utf-8 -*-

# Import relevant libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.mixture import GaussianMixture

# Read dataframes
train_x = pd.read_csv(r'train.csv')
train_y = pd.read_csv(r'trainLabels.csv')
test_x = pd.read_csv(r'test.csv')

# Convert dataframe columns into numpy arrays
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
test_x = np.asarray(test_x)
train_y = train_y.ravel()

print('training_x Shape:', train_x.shape, ',training_y Shape:', train_y.shape,
      ',testing_x Shape:', test_x.shape)

# Checking the models
all_x = np.r_[train_x, test_x]
print('all_x shape :', all_x.shape)

# Using the Gaussian Mixture model
lowest_bic = np.infty
bic = []
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=cv_type)
        gmm.fit(all_x)
        bic.append(gmm.aic(all_x))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

best_gmm.fit(all_x)
train_x = best_gmm.predict_proba(train_x)
test_x = best_gmm.predict_proba(test_x)

# Taking only two models to keep it simple
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

param_grid = dict()

# Grid search for best tuning parameters for KNN
grid_search_knn = GridSearchCV(knn, param_grid=param_grid, cv=10,
                               scoring='accuracy').fit(train_x, train_y)
print('best estimator KNN:', grid_search_knn.best_estimator_, 'Best Score',
      grid_search_knn.best_estimator_.score(train_x, train_y))
knn_best = grid_search_knn.best_estimator_

# Grid search for best tuning parameters for RandomForest
grid_search_rf = GridSearchCV(rf, param_grid=dict(), verbose=3,
                              scoring='accuracy', cv=10).fit(train_x, train_y)
print('best estimator RandomForest:', grid_search_rf.best_estimator_,
      'Best Score', grid_search_rf.best_estimator_.score(train_x, train_y))
rf_best = grid_search_rf.best_estimator_

knn_best.fit(train_x, train_y)
print(knn_best.predict(test_x)[0:10])
rf_best.fit(train_x, train_y)
print(rf_best.predict(test_x)[0:10])

# Scoring the models
print('Score for KNN :', cross_val_score(knn_best, train_x, train_y, cv=10,
      scoring='accuracy').mean())
print('Score for Random Forest :', cross_val_score(rf_best, train_x, train_y,
      cv=10, scoring='accuracy').max())

# Framing our solution
knn_best_pred = pd.DataFrame(knn_best.predict(test_x))
rf_best_pred = pd.DataFrame(rf_best.predict(test_x))

knn_best_pred.index += 1
rf_best_pred.index += 1

rf_best_pred.to_csv('Submission.csv')
