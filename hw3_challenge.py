# THIS HOMEWORK IS MY OWN WORK, WRITTEN WITHOUT COPYING FROM OTHER STUDENTS
# OR DIRECTLY FROM LARGE LANGUAGE MODELS SUCH AS CHATGPT.
# Any collaboration or external resources have been properly acknowledged.
# - The changes made to generate_feature_vector_challenge, impute_missing_values_challenge, and normalize_feature_matrix_challenge were written by me but the ideas for the changes were generated with the help of ChatGPT. I used ChatGPT to brainstorm ideas for how to improve the feature engineering and imputation steps, and then I implemented those ideas in code myself.
# Miguel_Chavez_Nava
# hw3_challenge.py

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn import metrics

from sklearn.linear_model import LogisticRegression, SGDClassifier

import hw3_main
from helper import *

def generate_feature_vector_challenge(df):
    static_variables = config['static']
    timeseries_variables = config['timeseries']
    feature_dict = {}
    for var in static_variables:
        data = df[df['Variable'] == var]['Value']
        if len(data) == 0 or data.values[0] == -1:
            value = np.nan
        else:
            value = data.values[0]
        if var == 'Gender':
            if value == 1:
                feature_dict.update({var + '_male': 1})
                feature_dict.update({var + '_female': 0})
                feature_dict.update({var + '_missing': 0})
            elif value == 0:
                feature_dict.update({var + '_male': 0})
                feature_dict.update({var + '_female': 1})
                feature_dict.update({var + '_missing': 0})
            else:
                feature_dict.update({var + '_male': 0})
                feature_dict.update({var + '_female': 0})
                feature_dict.update({var + '_missing': 1})
        elif var == 'ICUType':
            if np.isnan(value):
                for icu in [1, 2, 3, 4]:
                    feature_dict[f'ICUType_{icu}'] = 0
                feature_dict.update({var + '_missing': 1})
            else:
                for icu in [1, 2, 3, 4]:
                    feature_dict[f'ICUType_{icu}'] = 1 if value == icu else 0
                feature_dict.update({var + '_missing': 0})
        else: 
            if np.isnan(value):
                feature_dict.update({var: np.nan})
                feature_dict.update({var + '_missing': 1})
            else:
                feature_dict.update({var: data.values[0]})
                feature_dict.update({var + '_missing': 0})
    for var in timeseries_variables:
        data = df[df['Variable'] == var]['Value']
        if len(data) == 0 or data.values[0] == -1:
            feature_dict.update({var + '_mean': np.nan})
            feature_dict.update({var + '_std': np.nan})
            feature_dict.update({var + '_min': np.nan})
            feature_dict.update({var + '_max': np.nan})
            feature_dict.update({var + '_lastValue': np.nan})
            feature_dict.update({var + '_firstValue': np.nan})
            feature_dict.update({var + '_range': np.nan})
            feature_dict.update({var + '_count': 0})
            feature_dict.update({var + '_missing': 1})
        else:
            feature_dict.update({var + '_mean': data.mean()})
            feature_dict.update({var + '_std': data.std()})
            feature_dict.update({var + '_min': data.min()})
            feature_dict.update({var + '_max': data.max()})
            feature_dict.update({var + '_lastValue': data.values[-1]})
            feature_dict.update({var + '_firstValue': data.values[0]})
            feature_dict.update({var + '_range': data.max() - data.min()})
            feature_dict.update({var + '_count': len(data)})
            feature_dict.update({var + '_missing': 0})
    return feature_dict

def impute_missing_values_challenge(X):
    for col in range(X.shape[1]):
        for row in range(X.shape[0]):
            if np.isnan(X[row][col]):
                X[row][col] = np.nanmedian(X[:, col])
    return X

def normalize_feature_matrix_challenge(X):
    XMean = np.mean(X, axis=0)
    XStd = np.std(X, axis=0)
    XStd[XStd == 0] = 1
    XNorm = (X - XMean) / XStd
    return XNorm


def run_challenge(X_challenge, y_challenge, X_heldout):
    # Read challenge data
    # Train a linear classifier and apply to heldout dataset features
    # Use generate_challenge_labels to print the predicted labels

    C_range = np.logspace(-3, 3, 7)
    bestC = hw3_main.select_C(X_challenge, y_challenge, C_range, 'l2', 5, 'f1_score')
    clf = LogisticRegression(penalty='l2', C=bestC, max_iter=1000)### TODO: define your classifier with appropriate hyperparameters
    clf.fit(X_challenge, y_challenge)
    y_score = clf.predict_proba(X_heldout)[:, 1]
    y_label = clf.predict(X_heldout)
    confusionMatrix = metrics.confusion_matrix(y_challenge, clf.predict(X_challenge), labels=[-1, 1])
    print(confusionMatrix)
    make_challenge_submission(y_label, y_score)


if __name__ == '__main__':
    # Read challenge data
    X_challenge, y_challenge, X_heldout, feature_names = get_challenge_data()

    # TODO: Question 3: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels
    run_challenge(X_challenge, y_challenge, X_heldout)
    test_challenge_output()
