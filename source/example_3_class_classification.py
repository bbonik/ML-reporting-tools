#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vasileios vonikakis
@email: bbonik@gmail.com

Example of how to generate report for a multiclass classification model.
"""

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml_reporting_tools import generate_classification_report



# generate a simple binary classifier example

dataset = load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, 
    dataset.target, 
    test_size=0.2, 
    random_state=0
    )

classifier_object = RandomForestClassifier(
        n_estimators=100,
        criterion='gini',
        max_depth=10,
        random_state=0
        )

classifier_object.fit(X_train, y_train)
y_predict_proba = classifier_object.predict_proba(X_test)


# generate classification report
generate_classification_report(
        y_actual=y_test,
        y_predict_proba=y_predict_proba, 
        decision_threshold=0.5, 
        class_names_list=None,
        model_info='Random Forest 100 \n No cross-validation'
        )

