import os
import shutil

import numpy as np

from bot import bot_competition
from utils import *
from utils.vector_utils import tfidf_similarity, embedding_similarity, document_centroid, \
    similarity_to_centroid_tf_idf, similarity_to_centroid_semantic


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


# combine with other function
def generate_dataset(feature_vector, local_dir, rm_local_dir=True, use_raifer_data=True):
    document_workingset_file = local_dir + 'doc_ws.txt'
    doc_tfidf_dir = local_dir + 'doc_tf_idf/'
    index = local_dir + 'index'

    if use_raifer_data:
        trectext_file = raifer_trectext_file
        trec_reader = TrecReader(trec_file=raifer_trec_file)
    else:
        trectext_file = goren_trectext_file
        trec_reader = TrecReader(positions_file=goren_positions_file)

    trec_texts = utils.read_trectext_file(trectext_file)
    stopwords = open(stopwords_file).read().split('\n')[:-1]
    word_embedding_model = utils.load_word_embedding_model(embedding_model_file)

    lock.acquire()  # is this necessary?
    if not os.path.exists(doc_tfidf_dir):
        utils.ensure_dirs(local_dir)
        utils.create_index(trectext_file, new_index_name=index, indri_path=indri_path)
        utils.create_documents_workingset(document_workingset_file, ranked_lists=trec_reader)
        bot_competition.generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                                      new_index=index)
    lock.release()

    X = []
    Y = []
    for epoch in trec_reader.epochs():
        next_epoch = utils.get_next_epoch(epoch)
        if next_epoch not in trec_reader.epochs():
            break

        for qid in trec_reader.queries():
            doc_id = trec_reader[epoch][qid][0]
            next_doc_id = utils.get_next_doc_id(doc_id)
            query = utils.get_query_text(queries_file, qid)

            # create x
            X.append(extract_features(
                qid, epoch, query, trec_reader, trec_texts, doc_tfidf_dir, word_embedding_model, stopwords,
                feature_vector))

            # Create Y
            next_rank = trec_reader[next_epoch][qid].index(next_doc_id)
            Y.append(next_rank == 0)

    if rm_local_dir:
        shutil.rmtree(local_dir)
    return np.concatenate(X), np.stack(Y)


def extract_features(qid, epoch, query, trec_reader, trec_texts, doc_tfidf_dir, word_embedding_model, stopwords,
                     feature_vec=(False, False, False, False, True, False, False, False, False, False)):
    def tfidf_sim(x, y):
        return tfidf_similarity(doc_tfidf_dir + x, doc_tfidf_dir + y)

    def embed_sim(x, y):
        return embedding_similarity(trec_texts[x], trec_texts[y], word_embedding_model)

    def count_terms(x, t, opposite=False):
        return utils.count_occurrences(trec_texts[x], t, opposite, unique=False)

    def count_unique_terms(x, t, opposite=False):
        return utils.count_occurrences(trec_texts[x], t, opposite, unique=True)

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
