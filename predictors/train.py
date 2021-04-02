from copy import deepcopy

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

import winner_prediction
from utils import *


def parameter_search(algorithm, fixed_params, search_params, x, y):
    """
    This function performs hyperparameter optimization using grid search.
    This version uses cross validation, the original (from Raifer's paper) optimized on the train set.
    :param algorithm: Some machine learning algorithm
    :param fixed_params: Fixed model parameters
    :param search_params: Parameters which need to optimized
    :param x: The sample which are used to train and evaluate the weights (using cross validation)
    :param y: The labels which are used to train and evaluate the weights (using cross validation)
    :return: The model which received the highest cross validation score
    """
    parameter_dictionaries = utils.unpack_dictionary(search_params)

    best_score = 0
    best_model = None
    for params in parameter_dictionaries:
        model = algorithm(**params, **fixed_params)
        scores = cross_val_score(model, x, y, error_score='raise')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
    print(f'{best_model}: {best_score:.3f}')

    return best_model


def minmax_scale_queries(X_train, X_test, y_test):
    idx = pd.IndexSlice

    train_queries = X_train.index.get_level_values(1).unique()
    test_queries = X_test.index.get_level_values(1).unique()
    for query in train_queries:
        scaler = MinMaxScaler()
        X_train.loc[idx[:, query], :] = scaler.fit_transform(X_train.loc[idx[:, query], :])
        if query in test_queries:
            X_test.loc[idx[:, query], :] = scaler.transform(X_test.loc[idx[:, query], :])
        else:
            X_test.drop(index=query, level=1, inplace=True)
            y_test.drop(index=query, level=1, inplace=True)


def sub_sample(X: pd.DataFrame, y: pd.Series):
    value_counts = y.value_counts()
    n = min(value_counts)
    X_list, y_list = [], []
    for value in y.unique():
        index = y[y == value].index
        if value_counts[value] == n:
            X_list.append(X.loc[index])
            y_list.append(y.loc[index])
        else:
            sub_index = np.random.choice(index, size=n, replace=False)
            X_list.append(X.loc[sub_index])
            y_list.append(y.loc[sub_index])
    return pd.concat(X_list), pd.concat(y_list)


def naive_optimize(algorithm, parameter_grid, X, y):
    parameter_dictionaries = utils.unpack_dictionary(parameter_grid)

    best_score = 0
    best_model = None
    for params in parameter_dictionaries:
        model = deepcopy(algorithm)
        model.set_params(**params)
        model.fit(X, y)
        # score = model.score(X, y)
        predictions = winner_prediction.predict_winners(model, X)
        score = np.mean(predictions == y)
        if score > best_score:
            best_score = score
            best_model = model
    return best_model


def main():
    # parameter_grid = {'C': [1, 10, 50, 100], 'degree': [2, 3, 4, 5]}
    # algorithm = SVC(kernel='poly', probability=True)

    # parameter_grid = {'Classifier__C': [1, 10, 50, 100]}
    # algorithm = Pipeline([('Scaler', StandardScaler()),
    #                       ('Classifier', LogisticRegression(penalty='l1', solver='liblinear'))])

    parameter_grid = {'C': [1, 10, 50, 100]}
    algorithm = LogisticRegression(penalty='l1', solver='liblinear')

    # parameter_grid = {'n_estimators': [10, 50, 100, 500], 'max_leaf_nodes': [10, 20, 30]}
    # parameter_grid = {'n_estimators': [10, 50, 100, 500]}
    # algorithm = RandomForestClassifier()

    x, y = winner_prediction.generate_dataset(use_raifer_data=True)
    train_accuracies = []
    test_accuracies = []
    epochs = x.index.get_level_values(0).unique()
    for epoch in epochs:
        other_epochs = epochs.difference([epoch])
        X_train, X_test = x.loc[other_epochs], x.loc[[epoch]]
        y_train, y_test = y.loc[other_epochs], y.loc[[epoch]]
        # X_train, y_train = sub_sample(X_train, y_train)
        minmax_scale_queries(X_train, X_test, y_test)

        model = naive_optimize(algorithm, parameter_grid, X_train, y_train)
        # grid_search = GridSearchCV(estimator=algorithm, param_grid=parameter_grid).fit(X_train, y_train)
        # model = grid_search.best_estimator_

        # from collections import Counter
        # train_score = model.score(X_train, y_train)
        # train_counter = Counter(model.predict(X_train))
        # test_score = model.score(X_test, y_test)
        # test_counter = Counter(model.predict(X_test))

        predictions = winner_prediction.predict_winners(model, X_train)
        train_accuracies.append(np.mean(predictions == y_train))
        predictions = winner_prediction.predict_winners(model, X_test)
        test_accuracies.append(np.mean(predictions == y_test))
        
        # train_accuracies.append(model.score(X_train, y_train))
        # test_accuracies.append(model.score(X_test, y_test))
    print(f'The average train accuracy is {np.mean(train_accuracies):.3f}')
    print(train_accuracies)
    print(f'The average test accuracy is {np.mean(test_accuracies):.3f}')
    print(test_accuracies)


if __name__ == '__main__':
    main()
