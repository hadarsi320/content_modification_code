import os
import shutil

import numpy as np

import utils.general_utils as utils
from bot import bot_competition
from utils import *


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

    lock.acquire()
    if not os.path.exists(doc_tfidf_dir):
        utils.ensure_dirs(local_dir)
        utils.create_index(trectext_file, new_index_name=index, indri_path=indri_path)
        utils.create_documents_workingset(document_workingset_file, ranked_lists=trec_reader)
        bot_competition.generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                                      swig_path=swig_path, base_index=base_index, new_index=index)
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
            X.append(create_features(
                qid, epoch, query, trec_reader, trec_texts, doc_tfidf_dir, word_embedding_model, stopwords,
                feature_vector))

            # Create Y
            next_rank = trec_reader[next_epoch][qid].index(next_doc_id)
            Y.append(next_rank == 0)

    if rm_local_dir:
        shutil.rmtree(local_dir)
    return np.concatenate(X), np.stack(Y)
