import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import winner_prediction
from utils import *


def parameter_search(algorithm, fixed_params, search_params, x, y) -> sklearn.base.ClassifierMixin:
    """
    This function performs hyperparameter optimization using grid search.
    This version uses cross validation, the original (from Raifer's paper) optimized on the train set.
    @param algorithm: Some machine learning algorithm
    @param fixed_params: Fixed model parameters
    @param search_params: Parameters which need to optimized
    @param x: The sample which are used to train and evaluate the weights (using cross validation)
    @param y: The labels which are used to train and evaluate the weights (using cross validation)
    @return: The model which received the highest cross validation score
    """
    parameter_dictionaries = utils.unpack_dictionary(search_params)

    best_score = 0
    best_model = None
    for params in parameter_dictionaries:
        model = algorithm(**params, **fixed_params)

        # TODO check if using a pipeline is better
        # model = Pipeline([('Scaler', StandardScaler()), ('Classifier', algorithm(**params, **fixed_params))])

        scores = cross_val_score(model, x, y, error_score='raise')
        if scores.mean() > best_score:
            best_score = scores.mean()
            best_model = model
    print(f'{best_model}: {best_score:.3f}')

    return best_model


def minmax_scale_queries(train_set, test_set, test_labels):
    idx = pd.IndexSlice
    scalers = {}

    train_queries = train_set.index.get_level_values(1).unique()
    for query in train_queries:
        scaler = MinMaxScaler()
        train_set.loc[idx[:, query], :] = scaler.fit_transform(train_set.loc[idx[:, query], :])
        scalers[query] = scaler

    test_queries = test_set.index.get_level_values(1).unique()
    for query in test_queries:
        if query in scalers:
            scaler = scalers[query]
            test_set.loc[idx[:, query], :] = scaler.transform(test_set.loc[idx[:, query], :])
        else:
            test_set.drop(index=query, level=1, inplace=True)
            test_labels.drop(index=query, level=1, inplace=True)


def main():
    # parameters = {'C': [1, 10, 50, 100], 'degree': [2, 3, 4, 5]}
    # algorithm = SVC(kernel='poly', probability=True)

    parameters = {'Classifier__C': [1, 10, 50, 100]}
    algorithm = Pipeline([('Scaler', MinMaxScaler()),
                          ('Classifier', LogisticRegression(penalty='l1', solver='liblinear'))])

    x, y = winner_prediction.generate_dataset(use_raifer_data=True)
    epochs = x.index.get_level_values(0).unique()
    for epoch in epochs:
        train_x, train_y = x.loc[epochs.difference([epoch])], y.loc[epochs.difference([epoch])]
        test_x, test_y = x.loc[[epoch]], y.loc[[epoch]]
        minmax_scale_queries(train_x, test_x, test_y)

        grid_search = GridSearchCV(estimator=algorithm, param_grid=parameters).fit(train_x, train_y)
        model = grid_search.best_estimator_
        # print('Train score {:.3f} Test score {:.3f}'.format(model.score(train_x, train_y), model.score(test_x, test_y)))

        predictions = winner_prediction.predict_winners(model, test_x)
        accuracy = (predictions == test_y).mean()
        print(f'{accuracy:.3f}')


if __name__ == '__main__':
    main()
