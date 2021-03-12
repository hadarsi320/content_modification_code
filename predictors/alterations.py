import copy
import datetime
import pickle
import shutil
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from utils import *


def feature_selection(models, num_features, local_dir='alterations_tmp/', use_raifer_data=True, reverse=False):
    if reverse is False:
        features = [True] * num_features
        X, Y = generate_dataset(features, local_dir, use_raifer_data=use_raifer_data)
        accuracy = run_nested_cross_val(models, X, Y)  # baseline accuracy
        print(f'The baseline accuracy is {accuracy:.2%}')

        while sum(features) > 1:
            feature_acc = {}
            for i, feature in tqdm(enumerate(features), desc='Feature Combinations', total=len(features)):
                if feature:
                    new_features = copy.copy(features)
                    new_features[i] = False

                    X, Y = generate_dataset(new_features, local_dir, use_raifer_data=use_raifer_data)
                    feature_acc[i] = run_nested_cross_val(models, X, Y)

            disable_feature = max(feature_acc, key=lambda x: feature_acc[x])
            if feature_acc[disable_feature] > accuracy:
                features[disable_feature] = False
                accuracy = feature_acc[disable_feature]
                print(f'Disabled feature {disable_feature}, new accuracy is {accuracy:.2%}')
            else:
                print('No bad feature was found')
                break

    else:
        features = [False] * num_features
        accuracy = 0  # baseline accuracy
        print(f'The baseline accuracy is {accuracy:.2%} (since there are no features)')

        while sum(features) < num_features:
            feature_acc = {}
            for i, feature in tqdm(enumerate(features), desc='Feature Combinations', total=len(features)):
                if not feature:
                    new_features = copy.copy(features)
                    new_features[i] = True

                    X, Y = generate_dataset(new_features, local_dir, use_raifer_data=use_raifer_data)
                    feature_acc[i] = run_nested_cross_val(models, X, Y)

            disable_feature = max(feature_acc, key=lambda x: feature_acc[x])
            if feature_acc[disable_feature] > accuracy:
                features[disable_feature] = True
                accuracy = feature_acc[disable_feature]
                print(f'Enabled feature {disable_feature}, new accuracy is {accuracy:.2%}')
            else:
                print('No good feature was found')
                break

    shutil.rmtree(local_dir)
    return accuracy, features


@ignore_warnings(category=ConvergenceWarning)
def run_nested_cross_val(models, X, Y, random_state=35):
    pipelines = [Pipeline([('scaler', StandardScaler()), ('classifier', model)]) for model in models]
    kf = KFold(shuffle=True, random_state=random_state)

    acc = []
    for indices, test_indices in kf.split(X):
        X_, X_test = X[indices], X[test_indices]
        Y_, Y_test = Y[indices], Y[test_indices]

        val_acc = defaultdict(list)
        for train_indices, val_indices in kf.split(X_):
            X_train, X_val = X_[train_indices], X_[val_indices]
            Y_train, Y_val = Y_[train_indices], Y_[val_indices]

            for model in pipelines:
                model.fit(X_train, Y_train)
                val_acc[model].append(model.score(X_val, Y_val))
        best_model = max(val_acc, key=lambda key: np.average(val_acc[key]))
        best_model.fit(X_, Y_)
        acc.append(best_model.score(X_test, Y_test))
    return np.average(acc)


@ignore_warnings(category=ConvergenceWarning)
def run_cross_val(X, Y, model, n_splits=5, random_state=32):
    kf = KFold(shuffle=True, n_splits=n_splits, random_state=random_state)
    train_acc = []
    test_acc = []
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]

        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
        pipeline.fit(X_train, Y_train)

        train_acc.append(pipeline.score(X_train, Y_train))
        test_acc.append(pipeline.score(X_test, Y_test))
    return np.average(test_acc)


def select_model(models, X, Y):
    accuracy_dict = {model: run_cross_val(X, Y, model, n_splits=len(X)) for model in tqdm(models)}
    best_model = max(accuracy_dict, key=lambda k: accuracy_dict[k])
    pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', best_model)]).fit(X, Y)
    print(pipeline.score(X, Y))
    return pipeline


def main():
    local_dir = 'alterations_tmp/'
    models = [Perceptron(), GaussianNB(), BernoulliNB(),
              SVC(kernel='linear', probability=True), SVC(kernel='poly', probability=True),
              SVC(kernel='rbf', probability=True),
              LogisticRegression(penalty='l1', solver='liblinear'), LogisticRegression(penalty='l2'),
              LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga'),
              KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=5),
              KNeighborsClassifier(n_neighbors=10),
              DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=10),
              DecisionTreeClassifier(max_depth=None),
              RandomForestClassifier(max_depth=5), RandomForestClassifier(max_depth=10),
              RandomForestClassifier(max_depth=None), ]

    # # predict accuracy
    # accuracy, feature_vec = feature_selection(models, 10, use_raifer_data=False, reverse=False, local_dir=local_dir)
    # accuracy_, feature_vec_ = feature_selection(models, 10, use_raifer_data=False, reverse=True, local_dir=local_dir)
    # if accuracy_ > accuracy:
    #     accuracy, feature_vec = accuracy_, feature_vec_
    # print(f'The cross validated accuracy is {accuracy:.2%} with features {feature_vec}')

    # create model
    features_vec = (False, False, False, False, True, False, False, False, False, False)
    X, Y = generate_dataset(features_vec, local_dir, use_raifer_data=False)
    model = select_model(models, X, Y)
    print('The chosen model is:', model, sep='\n')
    pickle.dump(model, open('../rank_models/alteration_classifier.pkl', 'wb'))


if __name__ == '__main__':
    lock = Lock()

    start_time = datetime.datetime.now()
    main()
    print(f'\n\nTotal run time {datetime.datetime.now() - start_time}')
