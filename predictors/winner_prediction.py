import os
import re
import shutil
from typing import Iterable

import numpy as np
import pandas as pd

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
                query_terms = utils.get_query_text(queries_file, qid).split()

                item = dict(qid=qid, epoch=epoch, pid=pid)
                item['x'] = extract_features(
                    last_epoch, qid, pid, query_terms, trec_reader, trec_texts, doc_tfidf_dir, stopwords)
                item['y'] = trec_reader.get_top_player(epoch=epoch, qid=qid) == pid
                res.append(item)
        last_epoch = epoch

    shutil.rmtree(local_dir)

    res_df = pd.DataFrame(res).set_index(['epoch', 'qid', 'pid'])
    x = pd.DataFrame(res_df['x'].tolist(), index=res_df.index)  # unpacks the vector column
    y = res_df['y']
    return x, y


def term_changes(old_document: str, new_document: str, reference_document: str, target_terms: Iterable,
                 complement_terms=False):
    """
    Returns the micro features from Raifer's Paper (section 6.1) for some set of terms
    :param old_document: The version of the document in the last round
    :param new_document: The most recent version of the document in the last round
    :param reference_document: The document that the changes in our main document are compared to
    :param target_terms: The terms we focus on
    :param complement_terms: if true we look on all terms other than "terms"
    :return: The list [ADD(RD), ADD(~RD), RMV(RD),  RMV(~RD)]
    """
    cond = (lambda t, l: t in l) if not complement_terms else (lambda t, l: t not in l)

    new_document_ = re.sub("[^\w]", " ", new_document.lower())
    old_document_ = re.sub("[^\w]", " ", old_document.lower())
    reference_document_ = re.sub("[^\w]", " ", reference_document.lower())

    new_terms = set(new_document_.split())
    old_terms = set(old_document_.split())
    ref_terms = set(reference_document_.split())

    added_terms = new_terms - old_terms
    removed_terms = old_terms - new_terms

    res = [[] for _ in range(4)]
    for term in added_terms:
        if cond(term, target_terms):
            if term in ref_terms:
                res[0].append(term)
            else:
                res[1].append(term)

    for term in removed_terms:
        if cond(term, target_terms):
            if term in ref_terms:
                res[2].append(term)
            else:
                res[3].append(term)

    return [len(item) for item in res]


def extract_features(last_epoch, qid, pid, query_terms, trec_reader: TrecReader, trec_texts, doc_tfidf_dir, stopwords):
    def tfidf_sim(x, y):
        return tfidf_similarity(doc_tfidf_dir + x, doc_tfidf_dir + y)

    features = []

    prev_doc_id = utils.get_doc_id(last_epoch, qid, pid)
    doc_id = utils.get_next_doc_id(prev_doc_id)
    prev_winner_id = trec_reader[last_epoch][qid][0]
    assert prev_winner_id != prev_doc_id

    # Macro features
    features.append(tfidf_sim(doc_id, prev_doc_id))
    features.append(tfidf_sim(doc_id, prev_winner_id))
    features.append(tfidf_sim(prev_doc_id, prev_winner_id))

    # Micro features
    features.extend(term_changes(trec_texts[prev_doc_id], trec_texts[doc_id], trec_texts[prev_winner_id], query_terms))
    features.extend(term_changes(trec_texts[prev_doc_id], trec_texts[doc_id], trec_texts[prev_winner_id], stopwords))
    features.extend(term_changes(trec_texts[prev_doc_id], trec_texts[doc_id], trec_texts[prev_winner_id],
                                 set(query_terms + stopwords), complement_terms=True))
    return np.array(features)


def predict_winners(model, dataset: pd.DataFrame):
    predictions = []
    epochs = dataset.index.get_level_values(0).unique()
    for epoch in epochs:
        queries = dataset.loc[epoch].index.get_level_values(0).unique()
        for query in queries:
            x = dataset.loc[epoch, query]
            if hasattr(model, 'predict_proba'):
                scores = model.predict_proba(x)[:, 1]
            else:
                scores = model.decision_function(x)
            predictions.append(scores == max(scores))
    return pd.Series(np.concatenate(predictions), index=dataset.index)


if __name__ == '__main__':
    trec_reader = TrecReader(trec_file=raifer_trec_file)
    last_epoch = None
    for epoch in trec_reader.epochs():
        if last_epoch is None:
            last_epoch = epoch
            continue

        counter = 0
        for query in trec_reader.queries():
            if trec_reader.get_top_player(query, last_epoch) != trec_reader.get_top_player(query, epoch):
                counter += 1
        print(f'Epoch {epoch}: {counter}/{len(trec_reader.queries())}')
        last_epoch = epoch
