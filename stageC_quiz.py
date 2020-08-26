# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 08:49:38 2020

@author: Bambi
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv('Data_for_UCI_named.csv')

X = df.drop(columns=['stab', 'stabf'])
y = df['stabf']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
test_df = scaler.fit_transform(x_test)
x_trained = scaler.fit_transform(x_train)

pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier())])
pipe.fit(x_trained, y_train)
prediction = pipe.predict(test_df)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipe, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
print('Accuraccy:', np.mean(n_scores))

n_estimators = [50, 100, 300, 500, 1000]
min_samples_split = [2, 3, 5, 7, 9]
min_samples_leaf = [1, 2, 4, 6, 8]
max_features = ['auto', 'sqrt', 'log2', None] 
hyp = {'n_estimators': n_estimators,
       'min_samples_leaf': min_samples_leaf,
       'min_samples_split': min_samples_split,
       'max_features': max_features}


rf = ExtraTreesClassifier()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions = hyp, cv=5, n_iter=10, scoring = 'accuracy',
                   n_jobs = -1, verbose = 1, random_state=1)

# Fit the random search model
rf_random.fit(x_trained, y_train)
best_para = rf_random.best_params_
print('Best parameters are: ', best_para)

