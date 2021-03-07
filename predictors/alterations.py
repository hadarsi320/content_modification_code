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

import utils.general_utils as utils
from utils import *
from utils.vector_utils import tfidf_similarity, embedding_similarity, similarity_to_centroid_tf_idf, \
    document_centroid, similarity_to_centroid_semantic


def get_term_statistics(old_text, new_text, rival_text, target_terms):
    """
    Inspired by Raifer's paper
    """
    res = [0] * 4
    added_terms = utils.get_terms(new_text) - utils.get_terms(old_text)
    removed_terms = utils.get_terms(old_text) - utils.get_terms(new_text)

    for term in added_terms:
        if term in target_terms:
            if term in rival_text:
                res[0] += 1
            else:
                res[1] += 1

    for term in removed_terms:
        if term in target_terms:
            if term in rival_text:
                res[2] += 1
            else:
                res[3] += 1

    return res


def is_reliable(pid, qid, last_epoch, trec_reader, scaled=False):
    ranks = utils.get_player_ranks(last_epoch, qid, pid, trec_reader)

    # dsum = utils.discounted_sum(ranks)
    # foo = [utils.discounted_sum(utils.get_player_ranks(rival_pid, last_epoch, qid, trec_reader))
    #        for rival_pid in trec_reader.get_pids(qid) if rival_pid != pid]
    # bar = sum(dsum < rival_dsum for rival_dsum in foo)
    # return bar >= 0.5 * len(foo)

    if len(ranks) == 1:
        return True

    max_rank = len(trec_reader[last_epoch][qid]) - 1
    rank_change = utils.get_rank_change(ranks, max_rank, scaled)
    cum_rank_change = utils.dsum(rank_change)  # sum(rank_change)

    rivals_crc = [utils.dsum(
        utils.get_rank_change(
            utils.get_player_ranks(last_epoch, qid, rival_pid, trec_reader), max_rank, scaled))
        for rival_pid in trec_reader.get_pids(qid) if rival_pid != pid]
    amount_superior = sum((cum_rank_change > rival_rank_change for rival_rank_change in rivals_crc))
    return amount_superior >= 3


def create_features(qid, epoch, query, trec_reader, trec_texts, doc_tfidf_dir, word_embedding_model, stopwords,
                    feature_vec=(False, False, False, False, True, False, False, False, False, False)):
    def tfidf_sim(x, y):
        return tfidf_similarity(doc_tfidf_dir + x, doc_tfidf_dir + y)

    def embed_sim(x, y):
        return embedding_similarity(trec_texts[x], trec_texts[y], word_embedding_model)

    def count_terms(x, t, opposite=False):
        return utils.count_occurrences(trec_texts[x], t, opposite, False)

    def count_unique_terms(x, t, opposite=False):
        return utils.count_occurrences(trec_texts[x], t, opposite, True)

    query_words = query.split()
    old_doc_id = trec_reader[epoch][qid][0]
    new_doc_id = utils.get_next_doc_id(old_doc_id)
    features = []

    if feature_vec[0]:
        features.append(tfidf_sim(old_doc_id, new_doc_id))
        features.append(embed_sim(old_doc_id, new_doc_id))

    rival_doc_ids = [trec_reader[epoch][qid][i] for i in range(1, 3)]
    rivals_tfidf_centroid = document_centroid(doc_tfidf_dir, rival_doc_ids)
    if feature_vec[1]:
        features.append(similarity_to_centroid_tf_idf(doc_tfidf_dir + new_doc_id, rivals_tfidf_centroid))
        features.append(similarity_to_centroid_semantic(trec_texts[new_doc_id], trec_texts, rival_doc_ids,
                                                        word_embedding_model))
    if feature_vec[2]:
        features.append(similarity_to_centroid_tf_idf(doc_tfidf_dir + old_doc_id, rivals_tfidf_centroid))
        features.append(similarity_to_centroid_semantic(trec_texts[old_doc_id], trec_texts, rival_doc_ids,
                                                        word_embedding_model))

    rival_doc_ids = [trec_reader[epoch][qid][i] for i in range(1, 3)]
    for rival_doc_id in rival_doc_ids:
        rival_pid = utils.parse_doc_id(rival_doc_id)[2]
        reliable = is_reliable(rival_pid, qid, epoch, trec_reader, scaled=True)

        if feature_vec[3]:
            features.append(tfidf_sim(rival_doc_id, new_doc_id) * reliable)
            features.append(embed_sim(rival_doc_id, new_doc_id) * reliable)
        if feature_vec[4]:
            features.append(tfidf_sim(rival_doc_id, new_doc_id) * (1 - reliable))
            features.append(embed_sim(rival_doc_id, new_doc_id) * (1 - reliable))

        if feature_vec[5]:
            features.append(tfidf_sim(rival_doc_id, new_doc_id))
            features.append(embed_sim(rival_doc_id, new_doc_id))

    doc_pid = utils.parse_doc_id(old_doc_id)[2]
    player_acceleration = utils.get_player_accelerations(epoch, qid, trec_reader)
    if feature_vec[6]:
        if player_acceleration is not None and player_acceleration[0] != doc_pid:
            accel_pid = player_acceleration[0]
            accel_doc_id = utils.get_doc_id(epoch, qid, accel_pid)
            features.append(tfidf_sim(accel_doc_id, new_doc_id))
            features.append(embed_sim(accel_doc_id, new_doc_id))
        else:
            features += [0] * 2

    if feature_vec[7]:
        if player_acceleration is not None:
            decel_pid = player_acceleration[-1]
            decel_doc_id = utils.get_doc_id(epoch, qid, decel_pid)
            features.append(tfidf_sim(decel_doc_id, new_doc_id))
            features.append(embed_sim(decel_doc_id, new_doc_id))
        else:
            features += [0] * 2

    if feature_vec[8]:
        features.append(count_terms(new_doc_id, stopwords))
        features.append(count_terms(new_doc_id, query_words))
        features.append(count_terms(new_doc_id, stopwords + query_words, opposite=True))

    if feature_vec[9]:
        features.append(count_unique_terms(new_doc_id, stopwords))
        features.append(count_unique_terms(new_doc_id, query_words))
        features.append(count_unique_terms(new_doc_id, stopwords + query_words, opposite=True))

    return np.array(features).reshape((1, -1))


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
