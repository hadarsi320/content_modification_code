import os
import shutil

import numpy as np
import pandas as pd
import sklearn

from bot import bot_competition
from utils import *
from utils.vector_utils import tfidf_similarity


def generate_dataset(use_raifer_data, local_dir='local_dir/') -> (pd.DataFrame, pd.DataFrame):
    doc_tfidf_dir = local_dir + 'doc_tf_idf/'
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)

    if use_raifer_data:
        trectext_file = raifer_trectext_file
        trec_reader = TrecReader(trec_file=raifer_trec_file)
    else:
        trectext_file = goren_trectext_file
        trec_reader = TrecReader(positions_file=goren_positions_file)

    trec_texts = utils.read_trectext_file(trectext_file)
    stopwords = open(stopwords_file).read().split('\n')[:-1]
    bot_competition.create_doc_tfidf_files(local_dir, trectext_file, trec_reader, doc_tfidf_dir)

    res = []
    last_epoch = None
    for epoch in trec_reader.epochs():
        if last_epoch is None:
            last_epoch = epoch
            continue

        for qid in trec_reader.queries():
            if trec_reader.get_top_player(epoch=last_epoch, qid=qid) == \
                    trec_reader.get_top_player(epoch=epoch, qid=qid):
                continue

            for pid in trec_reader.get_pids(qid):
                if pid == trec_reader.get_top_player(epoch=last_epoch, qid=qid):
                    continue
                query_terms = utils.get_terms(utils.get_query_text(queries_file, qid))

                item = dict(qid=qid, epoch=epoch, pid=pid)
                item['x'] = extract_features(last_epoch, qid, pid, query_terms, trec_reader, trec_texts,
                                             doc_tfidf_dir, stopwords)
                item['y'] = trec_reader.get_top_player(epoch=epoch, qid=qid) == pid
                res.append(item)
        last_epoch = epoch

    shutil.rmtree(local_dir)

    res_df = pd.DataFrame(res) \
        .set_index(['epoch', 'qid', 'pid'])
    x = pd.DataFrame(res_df['x'].tolist(), index=res_df.index)
    y = res_df['y']
    return x, y


def term_changes(added_terms, removed_terms, reference_text, terms, complement_terms=False):
    res = [0] * 4

    for term in added_terms:
        if (not complement_terms and term in terms) or (complement_terms and term not in terms):
            if term in reference_text:
                res[0] += 1
            else:
                res[1] += 1

    for term in removed_terms:
        if (not complement_terms and term in terms) or (complement_terms and term not in terms):
            if term in reference_text:
                res[2] += 1
            else:
                res[3] += 1

    return res


def extract_features(epoch, qid, pid, query_terms, trec_reader: TrecReader, trec_texts, doc_tfidf_dir, stopwords):
    def tfidf_sim(x, y):
        return tfidf_similarity(doc_tfidf_dir + x, doc_tfidf_dir + y)

    features = []

    prev_doc_id = utils.get_doc_id(epoch, qid, pid)
    doc_id = utils.get_next_doc_id(prev_doc_id)
    prev_winner_id = trec_reader[epoch][qid][0]
    assert prev_winner_id != prev_doc_id

    # Macro features
    features.append(tfidf_sim(doc_id, prev_doc_id))
    features.append(tfidf_sim(doc_id, prev_winner_id))
    features.append(tfidf_sim(prev_doc_id, prev_winner_id))

    # Micro features
    removed_terms = utils.get_terms(trec_texts[prev_doc_id]) - utils.get_terms(trec_texts[doc_id])
    added_terms = utils.get_terms(trec_texts[doc_id]) - utils.get_terms(trec_texts[prev_doc_id])
    features.extend(term_changes(added_terms, removed_terms, trec_texts[prev_winner_id], query_terms))
    features.extend(term_changes(added_terms, removed_terms, trec_texts[prev_winner_id], stopwords))
    features.extend(term_changes(added_terms, removed_terms, trec_texts[prev_winner_id], query_terms.union(stopwords),
                                 complement_terms=True))
    return np.array(features)


def predict_winners(model, dataset: pd.DataFrame):
    predictions = []
    epochs = dataset.index.get_level_values(0).unique()
    for epoch in epochs:
        queries = dataset.loc[epoch].index.get_level_values(0).unique()
        for query in queries:
            x = dataset.loc[epoch, query]
            scores = model.decision_function(x)
            predictions.extend(scores == max(scores))
    return pd.Series(predictions, index=dataset.index)
