import shutil

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

import bot_competition
import readers
import utils
from utils import find_accelerating_player
from vector_functionality import tfidf_similarity, embedding_similarity


def term_difference(text_1, text_2, terms, opposite=False):
    return utils.count_occurrences(text_1, terms, opposite) - utils.count_occurrences(text_2, terms, opposite)


def term_analysis(old_text, new_text, rival_text, target_terms):
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


def create_features(qid, epoch, query, trec_reader, trec_texts, doc_tfidf_dir, word_embedding_model, stopwords):
    def tfidf_sim(x, y):
        return tfidf_similarity(doc_tfidf_dir + x, doc_tfidf_dir + y)

    def embed_sim(x, y):
        return embedding_similarity(trec_texts[x], trec_texts[y], word_embedding_model)

    def count_terms(x, t, opposite=False):
        return utils.count_occurrences(trec_texts[x], t, opposite)

    query_words = query.split()
    old_doc_id = trec_reader[epoch][qid][0]
    new_doc_id = utils.get_next_doc_id(old_doc_id)
    features = []

    # features.append(tfidf_sim(old_doc_id, new_doc_id))
    # features.append(embed_sim(old_doc_id, new_doc_id))

    # rival_doc_ids = [trec_reader[epoch][qid][i] for i in range(1, 3)]
    # rivals_tfidf_centroid = document_centroid(doc_tfidf_dir, rival_doc_ids)
    # features.append(similarity_to_centroid_tf_idf(doc_tfidf_dir + new_doc_id, rivals_tfidf_centroid))
    # features.append(similarity_to_centroid_semantic(trec_texts[new_doc_id], trec_texts, rival_doc_ids,
    #                                                 word_embedding_model))

    # rival_doc_ids = [trec_reader[epoch][qid][i] for i in range(1, 3)]
    # for rival_doc_id in rival_doc_ids:
    #     features.append(tfidf_sim(rival_doc_id, new_doc_id) - tfidf_sim(rival_doc_id, old_doc_id))
    #     features.append(embed_sim(rival_doc_id, new_doc_id) - embed_sim(rival_doc_id, old_doc_id))
    #     features.append(tfidf_sim(rival_doc_id, new_doc_id))
    #     features.append(embed_sim(rival_doc_id, new_doc_id))

    accel_pid = find_accelerating_player(trec_reader, qid, epoch)
    if accel_pid is not None:
        accel_doc_id = utils.get_doc_id(epoch, qid, accel_pid)
        features.append(tfidf_sim(accel_doc_id, new_doc_id))
        features.append(embed_sim(accel_doc_id, new_doc_id))
    else:
        features += [0] * 2

    features.append(count_terms(new_doc_id, stopwords))
    features.append(count_terms(new_doc_id, query_words))
    features.append(count_terms(new_doc_id, stopwords + query_words, True))

    return np.array(features)


# def cross_validate(models, X, Y):
#     X, X_test, Y, Y_test = train_test_split(X, Y, random_state=42)
#
#     kf = KFold(shuffle=True, random_state=54, n_splits=10)
#     results = {}
#     for model in models:
#
#         train_acc = []
#         val_acc = []
#         for train_indices, val_indices in kf.split(X):
#             X_train, X_val = X[train_indices], X[val_indices]
#             Y_train, Y_val = Y[train_indices], Y[val_indices]
#
#             # scaler = StandardScaler()
#             # scaler.fit(X_train)
#             # X_train = scaler.transform(X_train)
#             # X_val = scaler.transform(X_val)
#             #
#             # model.fit(X_train, Y_train)
#             # train_acc.append(model.score(X_train, Y_train))
#             # val_acc.append(model.score(X_val, Y_val))
#
#             pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
#             pipeline.fit(X_train, Y_train)
#             train_acc.append(pipeline.score(X_train, Y_train))
#             val_acc.append(pipeline.score(X_val, Y_val))
#
#         scaler = StandardScaler()
#         scaler.fit(X)
#         X_scaled = scaler.transform(X)
#         X_test_scaled = scaler.transform(X_test)
#         model.fit(X_scaled, Y)
#         test_acc = model.score(X_test_scaled, Y_test)
#
#         results[str(model)] = np.average(train_acc), np.average(val_acc), test_acc
#
#     return results


def generate_learning_dataset():
    embedding_model_file = '/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model'
    base_index = '/lv_local/home/hadarsi/work_files/clueweb_index/'
    swig_path = '/lv_local/home/hadarsi/indri-5.6/swig/obj/java/'
    trec_file = 'data/trec_file_original_sorted.txt'
    indri_path = '/lv_local/home/hadarsi/indri/'
    queries_file = 'data/queries_seo_exp.xml'
    trectext_file = 'data/documents.trectext'
    stopwords_file = 'data/stopwords_list'

    local_dir = 'tmp/'
    document_workingset_file = local_dir + 'doc_ws'
    doc_tfidf_dir = local_dir + 'doc_tf_idf/'
    index = local_dir + 'index'

    trec_reader = readers.TrecReader(trec_file)
    trec_texts = utils.read_trectext_file(trectext_file)
    stopwords = open(stopwords_file).read().split('\n')[:-1]

    utils.ensure_dirs(local_dir)
    utils.create_index(trectext_file, new_index_name=index, indri_path=indri_path)
    utils.create_documents_workingset(document_workingset_file, ranked_lists=trec_reader)
    bot_competition.generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                                  swig_path=swig_path, base_index=base_index, new_index=index)
    word_embedding_model = utils.load_word_embedding_model(embedding_model_file)

    X = []
    Y = []
    for epoch in trec_reader.get_epochs():
        next_epoch = utils.get_next_epoch(epoch)
        if next_epoch not in trec_reader.get_epochs():
            break

        for qid in trec_reader.get_queries():
            doc_id = trec_reader[epoch][qid][0]
            next_doc_id = utils.get_next_doc_id(doc_id)
            query = utils.get_query_text(queries_file, qid)

            # create x
            X.append(create_features(qid, epoch, query, trec_reader, trec_texts, doc_tfidf_dir, word_embedding_model,
                                     stopwords))

            # Create Y
            next_rank = trec_reader[next_epoch][qid].index(next_doc_id)
            Y.append(next_rank == 0)

    shutil.rmtree(local_dir)
    return np.array(X), np.array(Y)


def test_models(models, X, Y):
    # results = cross_validate(models, X, Y)
    kf = KFold(shuffle=True, random_state=54, n_splits=25)
    results = {}
    for model in models:
        train_acc = []
        test_acc = []
        for train_indices, test_indices in kf.split(X):
            X_train, X_test = X[train_indices], X[test_indices]
            Y_train, Y_test = Y[train_indices], Y[test_indices]

            pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', model)])
            pipeline.fit(X_train, Y_train)
            train_acc.append(pipeline.score(X_train, Y_train))
            test_acc.append(pipeline.score(X_test, X_test))

        results[str(model)] = np.average(train_acc), np.average(test_acc)

    top_models = sorted(results, key=lambda key: results[key][1], reverse=True)[:5]
    for i, model in enumerate(top_models):
        print(f'{str(i + 1) + ".":3} {model:75} '
              f'train: {results[model][0]:.3f}\ttest: {results[model][1]:.3f}')


def run_kfold_cross_val(X, Y, model, n_splits=10):
    kf = KFold(shuffle=True, random_state=42, n_splits=n_splits)
    test_acc = []
    for train_indices, test_indices in kf.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        Y_train, Y_test = Y[train_indices], Y[test_indices]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, Y_train)
        test_acc.append(model.score(X_test, Y_test))
    return np.average(test_acc)


def train_alteration_classifier(X, Y, model=RandomForestClassifier, model_params=None):
    if model_params is not None:
        accuracy_dict = {}
        for i, kwargs in enumerate(model_params):
            classifier = model(**kwargs)
            accuracy_dict[i] = run_kfold_cross_val(X, Y, classifier)

        max_index = max(accuracy_dict, key=lambda x: accuracy_dict[x])
        max_kwargs = model_params[max_index]

    else:
        max_kwargs = {}

    classifier = Pipeline([('scaler', StandardScaler()),
                           ('classifier', model(**max_kwargs))]).fit(X, Y)
    return classifier


def main():
    classifiers_dir = 'classifiers/'
    utils.ensure_dirs(classifiers_dir)

    X, Y = generate_learning_dataset()
    print('Features created')

    params = [{'max_depth': 5}, {'max_depth': 10}, {'max_depth':20}, {'max_depth': 50}, {'max_depth': None}, ]
    alter_classifier = train_alteration_classifier(X, Y, model_params=params)
    pickle.dump(alter_classifier, open(f'{classifiers_dir}/alteration_classifier.pkl', 'wb'))

    # models = [Perceptron(), GaussianNB(), BernoulliNB(), SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf'),
    #           LogisticRegression(penalty='l1', solver='liblinear'), LogisticRegression(penalty='l2'),
    #           LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga'),
    #           KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=5),
    #           KNeighborsClassifier(n_neighbors=10),
    #           DecisionTreeClassifier(max_depth=5), DecisionTreeClassifier(max_depth=10),
    #           DecisionTreeClassifier(max_depth=None),
    #           RandomForestClassifier(max_depth=5), RandomForestClassifier(max_depth=10),
    #           RandomForestClassifier(max_depth=None), ]
    # test_models(models, X, Y)


if __name__ == '__main__':
    main()
