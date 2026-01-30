import os
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import copy

def train_best_xgboost_model(X_train, y_train, X_val, y_val, X_test, y_test, params):
    """
    Train an XGBoost model with the best hyperparameters and evaluate it on the test set.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.
    params (dict): Best hyperparameters for the XGBoost model.

    Returns:
    float: Accuracy of the model on the test set.
    """
    # Train initial predictor
    best_val = 0
    for i in range(1, 12):
        clf = XGBClassifier(max_depth = i, tree_method = 'hist', random_state = 0, seed = 0, device = 'cuda')
        clf.fit(X_train, y_train)
        xtrain_pred = clf.predict(X_train)
        xval_pred = clf.predict(X_val)
        xtest_pred = clf.predict(X_test)
        train_acc = accuracy_score(xtrain_pred, y_train)*100
        val_acc = accuracy_score(xval_pred, y_val)*100
        test_acc = accuracy_score(xtest_pred, y_test)*100
        if val_acc > best_val:
            best_train, best_val, best_test = train_acc, val_acc, test_acc
            best_clf = copy.deepcopy(clf)

    return best_clf, best_train, best_val, best_test

def train_xgboost_model(X_train, y_train, X_val, y_val, X_test, y_test, params):
    """
    Train an XGBoost model with the best hyperparameters and evaluate it on the test set.

    Parameters:
    X_train (pd.DataFrame): Training features.
    y_train (pd.Series): Training target.
    X_val (pd.DataFrame): Validation features.
    y_val (pd.Series): Validation target.
    X_test (pd.DataFrame): Test features.
    y_test (pd.Series): Test target.
    params (dict): Best hyperparameters for the XGBoost model.

    Returns:
    float: Accuracy of the model on the test set.
    """
    # Train initial predictor
    # best_val = 0
    clf = XGBClassifier(random_state = 0, seed = 0, tree_method="gpu_hist", device = 'cuda')
    clf.fit(X_train, y_train)
    xtrain_pred = clf.predict(X_train)
    xval_pred = clf.predict(X_val)
    xtest_pred = clf.predict(X_test)
    train_acc = accuracy_score(xtrain_pred, y_train)*100
    val_acc = accuracy_score(xval_pred, y_val)*100
    test_acc = accuracy_score(xtest_pred, y_test)*100

    return clf, train_acc, val_acc, test_acc