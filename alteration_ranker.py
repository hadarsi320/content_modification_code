import os
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import bot_competition
import readers
import utils
from bot_competition import find_accelerating_document
from vector_functionality import tfidf_similarity, embedding_similarity


def term_difference(text_1, text_2, terms, opposite=False):
    return utils.count_occurrences(text_1, terms, opposite) - utils.count_occurrences(text_2, terms, opposite)


def create_features(qid, epoch, trec_texts, trec_reader, doc_tfidf_dir, word_embedding_model, stopwords, query):
    query_words = query.split()
    doc_id = trec_reader[epoch][qid][0]
    next_doc_id = utils.get_next_doc_id(doc_id)
    features = [tfidf_similarity(doc_tfidf_dir + doc_id, doc_tfidf_dir + next_doc_id),
                embedding_similarity(trec_texts[doc_id], trec_texts[next_doc_id], word_embedding_model),
                term_difference(trec_texts[next_doc_id], trec_texts[doc_id], stopwords),
                term_difference(trec_texts[next_doc_id], trec_texts[doc_id], query_words),
                term_difference(trec_texts[next_doc_id], trec_texts[doc_id], stopwords + query_words, opposite=True)]

    for i in range(1, 3):
        rival_doc_id = trec_reader[epoch][qid][i]
        features.append(tfidf_similarity(doc_tfidf_dir + rival_doc_id, doc_tfidf_dir + next_doc_id))
        features.append(embedding_similarity(trec_texts[rival_doc_id], trec_texts[next_doc_id],
                                             word_embedding_model))

    accel_doc = find_accelerating_document
    # features.append(term_difference(trec_texts[next_doc_id], trec_texts[rival_doc_id], stopwords))
    # features.append(term_difference(trec_texts[next_doc_id], trec_texts[rival_doc_id], query_words))
    # features.append(term_difference(trec_texts[next_doc_id], trec_texts[rival_doc_id], stopwords + query_words,
    #                                 opposite=True))

    return features


def main():
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

    os.makedirs(local_dir)
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
            X.append(create_features(qid, epoch, trec_texts, trec_reader, doc_tfidf_dir, word_embedding_model,
                                     stopwords, query))

            # Create Y
            next_rank = trec_reader[next_epoch][qid].index(next_doc_id)
            Y.append(next_rank == 0)

    shutil.rmtree(local_dir)
    X = np.array(X)
    Y = np.array(Y)

    models = [Perceptron(), LogisticRegression(), GaussianNB(), BernoulliNB(), SVC(kernel='linear'), SVC(kernel='poly'),
              SVC(kernel='rbf'), KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=5),
              KNeighborsClassifier(n_neighbors=10), DecisionTreeClassifier(max_depth=5),
              DecisionTreeClassifier(max_depth=10), DecisionTreeClassifier(max_depth=None)]

    kf = KFold(shuffle=True, random_state=54, n_splits=10)
    for model in models:
        acc = []
        for train_indices, test_indices in kf.split(X):
            scaler = StandardScaler()
            X_train, X_test = X[train_indices], X[test_indices]
            Y_train, Y_test = Y[train_indices], Y[test_indices]

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, Y_train)
            acc.append(model.score(X_test, Y_test))

        print(f'{str(model):60} {np.average(acc):.3f}')


if __name__ == '__main__':
    main()
