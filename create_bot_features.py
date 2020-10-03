import logging
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import cpu_count

import numpy as np
from deprecated import deprecated
from nltk import sent_tokenize

from gen_utils import run_bash_command, list_multiprocessing, run_and_print
from utils import clean_texts, get_java_object, create_trectext_file, create_index, \
    run_model, create_features_file_diff, read_raw_trec_file, create_trec_eval_file, order_trec_file, retrieve_scores, \
    transform_query_text, read_queries_file, get_query_text, reverese_query, create_index_to_query_dict, \
    generate_pair_name, ensure_dir, tokenize_document, is_file_empty, get_next_doc_id
from vector_functionality import query_term_freq, embedding_similarity, calculate_similarity_to_docs_centroid_tf_idf, \
    document_centroid, calculate_semantic_similarity_to_top_docs, get_text_centroid, add_dict, cosine_similarity


def create_sentence_pairs(top_docs, ref_doc, texts):
    result = {}
    ref_sentences = tokenize_document(texts[ref_doc])  # sent_tokenize(texts[ref_doc])  # texts[ref_doc].split('\n')
    for doc in top_docs:
        doc_sentences = tokenize_document(texts[doc])  # sent_tokenize(texts[doc])  # texts[doc].split('\n')
        for in_index, top_sentence in enumerate(doc_sentences):
            # if top_sentence.replace("\n", " ").rstrip() in ref_sentences:
            if top_sentence in ref_sentences:
                continue
            for out_index, ref_sentence in enumerate(ref_sentences):
                key = ref_doc + "$" + doc + "_" + str(out_index) + "_" + str(in_index)
                result[key] = ref_sentence + "\t" + top_sentence
    return result


def create_raw_dataset(ranked_lists, doc_texts, output_file, ref_index, top_docs_index, current_epoch=None,
                       current_qid=None):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as output:
        for epoch in ranked_lists:
            if current_epoch and epoch != current_epoch:
                continue
            for query in ranked_lists[epoch]:
                if current_qid and query != current_qid:
                    continue
                top_docs = ranked_lists[epoch][query][:top_docs_index]
                ref_doc = ranked_lists[epoch][query][ref_index]
                pairs = create_sentence_pairs(top_docs, ref_doc, doc_texts)
                for key in pairs:
                    output.write(str(int(query)) + str(epoch) + "\t" + key + "\t" + pairs[key] + "\n")


def read_raw_ds(raw_dataset):
    result = defaultdict(dict)
    with open(raw_dataset, encoding="utf-8") as ds:
        for line in ds:
            query = line.split("\t")[0]
            key = line.split("\t")[1]
            sentence_out = line.split("\t")[2]
            sentence_in = line.split("\t")[3].rstrip()
            result[query][key] = {"in": sentence_in, "out": sentence_out}
    return dict(result)


def context_similarity(replacement_index, ref_sentences, sentence_compared, mode, model, stemmer=None):
    if mode == "own":
        ref_sentence = ref_sentences[replacement_index]
        return embedding_similarity(clean_texts(ref_sentence), clean_texts(sentence_compared), model, stemmer)
    if mode == "pred":
        if replacement_index + 1 == len(ref_sentences):
            sentence = ref_sentences[replacement_index]
        else:
            sentence = ref_sentences[replacement_index + 1]
        return embedding_similarity(clean_texts(sentence), clean_texts(sentence_compared), model, stemmer)
    if mode == "prev":
        if replacement_index == 0:
            sentence = ref_sentences[replacement_index]
        else:
            sentence = ref_sentences[replacement_index - 1]
        return embedding_similarity(clean_texts(sentence), clean_texts(sentence_compared), model, stemmer)


def get_past_winners(ranked_lists, epoch, query):
    past_winners = []
    for iteration in range(int(epoch)):
        current_epoch = str(iteration + 1).zfill(2)
        past_winners.append(ranked_lists[current_epoch][query][0])
    return past_winners


def create_weighted_dict(base_dict, weight):
    result = {}
    for token in base_dict:
        result[token] = float(base_dict[token]) * weight
    return result


def get_past_winners_tfidf_centroid(past_winners, document_vectors_dir):
    result = {}
    decay_factors = [0.01 * math.exp(-0.01 * (len(past_winners) - i)) for i in range(len(past_winners))]
    denominator = sum(decay_factors)
    for i, doc in enumerate(past_winners):
        doc_tfidf = get_java_object(document_vectors_dir + doc)
        decay = decay_factors[i] / denominator
        normalized_vector = create_weighted_dict(doc_tfidf, decay)
        result = add_dict(result, normalized_vector)
    return result


def past_winners_centroid(past_winners, texts, model, stemmer=None):
    sum_vector = None
    decay_factors = [0.01 * math.exp(-0.01 * (len(past_winners) - i)) for i in range(len(past_winners))]
    denominator = sum(decay_factors)
    for i, doc in enumerate(past_winners):
        text = texts[doc]
        vector = get_text_centroid(clean_texts(text), model, stemmer)
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector += vector * decay_factors[i] / denominator
    return sum_vector


def write_files(feature_list, feature_vals, output_dir, qrid, ref):
    epoch, qid = reverese_query(qrid)
    # ind_name = {-1: "5", 1: "2"}
    # query_write = qid + epoch.lstrip('0') + ind_name[ref]
    query_write = f'{qid}{epoch.lstrip("0")}{ref+1}'
    for feature in feature_list:
        with open(output_dir + "doc" + feature + "_" + query_write, 'w') as out:
            for pair in feature_vals[feature]:
                name = generate_pair_name(pair)
                value = feature_vals[feature][pair]
                out.write(name + " " + str(value) + "\n")


def create_features_new(raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir,
                        sentence_tfidf_vectors_dir, query_text, output_dir, qrid, word_embed_model):
    feature_vals = defaultdict(dict)
    relevant_pairs = raw_ds[qrid]
    epoch, qid = reverese_query(qrid)
    query_text = clean_texts(query_text)
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]

    past_winners = get_past_winners(ranked_lists, epoch, qid)
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners, doc_texts, word_embed_model, True)
    past_winners_tfidf_centroid_vector = get_past_winners_tfidf_centroid(past_winners, doc_tfidf_vectors_dir)
    top_docs = ranked_lists[epoch][qid][:top_doc_index]
    ref_doc = ranked_lists[epoch][qid][ref_doc_index]
    ref_sentences = sent_tokenize(doc_texts[ref_doc])  # doc_texts[ref_doc].split('\n')
    top_docs_tfidf_centroid = document_centroid([get_java_object(doc_tfidf_vectors_dir + doc) for doc in top_docs])
    for pair in relevant_pairs:
        # Sentences have been cleaned
        sentence_in = relevant_pairs[pair]["in"]
        sentence_out = relevant_pairs[pair]["out"]
        in_vec = get_text_centroid(sentence_in, word_embed_model, True)
        out_vec = get_text_centroid(sentence_out, word_embed_model, True)
        replace_index = int(pair.split("_")[1])

        feature_vals['FractionOfQueryWordsIn'][pair] = query_term_freq("avg", sentence_in, query_text)
        feature_vals['FractionOfQueryWordsOut'][pair] = query_term_freq("avg", sentence_out, query_text)
        feature_vals['CosineToCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            sentence_tfidf_vectors_dir + pair.split("$")[1].split("_")[0] + "_" +
            pair.split("_")[2], top_docs_tfidf_centroid)
        feature_vals['CosineToCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            sentence_tfidf_vectors_dir + pair.split("$")[0] + "_" + pair.split("_")[1], top_docs_tfidf_centroid)
        feature_vals["CosineToCentroidInVec"][pair] = \
            calculate_semantic_similarity_to_top_docs(sentence_in, top_docs, doc_texts, word_embed_model, True)
        feature_vals["CosineToCentroidOutVec"][pair] = \
            calculate_semantic_similarity_to_top_docs(sentence_out, top_docs, doc_texts, word_embed_model, True)
        feature_vals['CosineToWinnerCentroidInVec'][pair] = \
            cosine_similarity(in_vec, past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidOutVec'][pair] = \
            cosine_similarity(out_vec, past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            sentence_tfidf_vectors_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2],
            past_winners_tfidf_centroid_vector)
        feature_vals['CosineToWinnerCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            sentence_tfidf_vectors_dir + pair.split("$")[0] + "_" + pair.split("_")[1],
            past_winners_tfidf_centroid_vector)
        feature_vals['SimilarityToPrev'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_in, "prev", word_embed_model, True)
        feature_vals['SimilarityToRefSentence'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_in, "own", word_embed_model, True)
        feature_vals['SimilarityToPred'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_in, "pred", word_embed_model, True)
        feature_vals['SimilarityToPrevRef'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_out, "prev", word_embed_model, True)
    write_files(feature_list, feature_vals, output_dir, qrid, ref_doc_index)


def create_features_og(raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir,
                       tfidf_sentence_dir, queries, output_dir, qrid):
    global word_embd_model
    feature_vals = defaultdict(dict)
    relevant_pairs = raw_ds[qrid]
    epoch, qid = reverese_query(qrid)
    query_text = clean_texts(queries[qrid])
    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef",
                    "SimilarityToPredRef"]

    past_winners = get_past_winners(ranked_lists, epoch, qid)
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners, doc_texts, word_embd_model, True)
    past_winners_tfidf_centroid_vector = get_past_winners_tfidf_centroid(past_winners, doc_tfidf_vectors_dir)
    top_docs = ranked_lists[epoch][qid][:top_doc_index]
    ref_doc = ranked_lists[epoch][qid][ref_doc_index]
    ref_sentences = sent_tokenize(doc_texts[ref_doc])  # doc_texts[ref_doc].split('\n')
    top_docs_tfidf_centroid = document_centroid([get_java_object(doc_tfidf_vectors_dir + doc) for doc in top_docs])
    for pair in relevant_pairs:
        sentence_in = relevant_pairs[pair]["in"]
        sentence_out = relevant_pairs[pair]["out"]
        in_vec = get_text_centroid(clean_texts(sentence_in), word_embd_model, True)
        out_vec = get_text_centroid(clean_texts(sentence_out), word_embd_model, True)
        replace_index = int(pair.split("_")[1])

        feature_vals['FractionOfQueryWordsIn'][pair] = query_term_freq("avg", sentence_in, query_text)
        feature_vals['FractionOfQueryWordsOut'][pair] = query_term_freq("avg", sentence_out, query_text)
        feature_vals['CosineToCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2], top_docs_tfidf_centroid)
        feature_vals['CosineToCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[0] + "_" + pair.split("_")[1], top_docs_tfidf_centroid)
        feature_vals["CosineToCentroidInVec"][pair] = calculate_semantic_similarity_to_top_docs(sentence_in, top_docs,
                                                                                                doc_texts,
                                                                                                word_embd_model, True)
        feature_vals["CosineToCentroidOutVec"][pair] = calculate_semantic_similarity_to_top_docs(sentence_out, top_docs,
                                                                                                 doc_texts,
                                                                                                 word_embd_model, True)
        feature_vals['CosineToWinnerCentroidInVec'][pair] = cosine_similarity(in_vec,
                                                                              past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidOutVec'][pair] = cosine_similarity(out_vec,
                                                                               past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidIn'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2],
            past_winners_tfidf_centroid_vector)
        feature_vals['CosineToWinnerCentroidOut'][pair] = calculate_similarity_to_docs_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[0] + "_" + pair.split("_")[1], past_winners_tfidf_centroid_vector)
        feature_vals['SimilarityToPrev'][pair] = context_similarity(replace_index, ref_sentences, sentence_in, "prev",
                                                                    word_embd_model, True)
        feature_vals['SimilarityToRefSentence'][pair] = context_similarity(replace_index, ref_sentences, sentence_in,
                                                                           "own", word_embd_model, True)
        feature_vals['SimilarityToPred'][pair] = context_similarity(replace_index, ref_sentences, sentence_in, "pred",
                                                                    word_embd_model, True)
        feature_vals['SimilarityToPrevRef'][pair] = context_similarity(replace_index, ref_sentences, sentence_out,
                                                                       "prev", word_embd_model, True)
    write_files(feature_list, feature_vals, output_dir, qrid, ref_doc_index)


def feature_creation_parallel(raw_dataset_file, ranked_lists, doc_texts, top_doc_index, ref_doc_index,
                              doc_tfidf_vectors_dir, tfidf_sentence_dir, queries, output_feature_files_dir,
                              output_final_features_dir, workingset_file):
    global word_embd_model
    args = [qid for qid in queries]
    if not os.path.exists(output_feature_files_dir):
        os.makedirs(output_feature_files_dir)
    if not os.path.exists(output_final_features_dir):
        os.makedirs(output_final_features_dir)
    raw_ds = read_raw_ds(raw_dataset_file)
    create_ws(raw_ds, workingset_file, ref_doc_index)
    func = partial(create_features_og, raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index,
                   doc_tfidf_vectors_dir, tfidf_sentence_dir, queries, output_feature_files_dir)
    workers = cpu_count() - 1
    list_multiprocessing(args, func, workers=workers)
    command = "perl scripts/generateSentences.pl " + output_feature_files_dir + " " + workingset_file
    run_bash_command(command)
    run_bash_command("mv features " + output_final_features_dir)


def feature_creation_single(raw_dataset_file, ranked_lists, doc_texts, ref_doc_index, top_doc_index,
                            doc_tfidf_vectors_dir, sentence_tfidf_vectors_dir, qrid, query_text,
                            output_feature_files_dir, output_final_features_file, workingset_file, word_embed_model):
    logger = logging.getLogger(sys.argv[0])
    if not os.path.exists(output_feature_files_dir):
        os.makedirs(output_feature_files_dir)
    ensure_dir(output_final_features_file)
    raw_ds = read_raw_ds(raw_dataset_file)
    create_ws(raw_ds, workingset_file, ref_doc_index)
    create_features_new(raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir,
                        sentence_tfidf_vectors_dir, query_text, output_feature_files_dir, qrid, word_embed_model)
    command = "perl scripts/generateSentences.pl " + output_feature_files_dir + " " + workingset_file
    logger.info(command)
    run_bash_command(command)
    command = "mv features " + output_final_features_file
    logger.info(command)
    run_bash_command(command)
    print()


def run_svm_rank_model(test_file, model_file, predictions_folder):
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    predictions_file = predictions_folder + os.path.basename(model_file)
    command = "./svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    run_and_print(command, command_name='Ranking')
    return predictions_file


def create_index_to_doc_name_dict(features):
    doc_name_index = {}
    index = 0
    with open(features) as ds:
        for line in ds:
            rec = line.split("# ")
            doc_name = rec[1].rstrip()
            doc_name_index[index] = doc_name
            index += 1
    return doc_name_index


def create_sentence_vector_files(output_dir, raw_ds_file, base_index, new_index, swig_path, documents_ws):
    for index in [base_index, new_index]:
        if not os.path.exists(index):
            raise ValueError('The index {} does not exist'.format(index))

    # command = f'java -Djava.library.path={swig_path} -cp seo_indri_utils.jar PrepareTFIDFVectorsSentences ' \
    #           f'{index_path} {raw_ds_file} {output_dir}'
    command = f'java -Djava.library.path={swig_path} -cp seo_indri_utils.jar PrepareTFIDFVectorsSentences ' \
              f'{base_index} {new_index} {raw_ds_file} {output_dir} {documents_ws}'
    run_and_print(command, command_name='PrepareTFIDFVectorsSentences')


def update_text_doc(text, new_sentence, replacement_index):
    sentences = sent_tokenize(text)  # text.split('\n')
    sentences[replacement_index] = new_sentence
    return "\n".join(sentences)


def update_texts(doc_texts, pairs_ranked_lists, sentence_data):
    new_texts = {}
    for qid in pairs_ranked_lists:
        chosen_pair = pairs_ranked_lists[qid][0]
        ref_doc = chosen_pair.split("$")[0]
        replacement_index = int(chosen_pair.split("_")[1])
        sentence_in = sentence_data[qid][chosen_pair]["in"]
        new_texts[ref_doc] = update_text_doc(doc_texts[ref_doc], sentence_in, replacement_index)
    for doc in doc_texts:
        if doc not in new_texts:
            new_texts[doc] = doc_texts[doc]
    return new_texts


def create_ws(raw_ds, ws_fname, ref):
    # ind_name = {-1: "5", 1: "2"}
    ensure_dir(ws_fname)
    with open(ws_fname, 'w') as ws:
        for qrid in raw_ds:
            epoch, qid = reverese_query(qrid)
            # query_write = qid + str(int(epoch)) + ind_name[ref]
            query_write = f'{qid}{epoch.lstrip("0")}{ref+1}'
            for i, pair in enumerate(raw_ds[qrid]):
                name = generate_pair_name(pair)
                ws.write(query_write + " Q0 " + name + " 0 " + str(i + 1) + " pairs_seo\n")


def create_new_trectext(doc, texts, new_text, new_trectext_name):
    text_copy = deepcopy(texts)
    text_copy[doc] = new_text
    create_trectext_file(text_copy, new_trectext_name, "ws_debug")


def create_reranking_ws(qrid, ranked_list, file_name):
    with open(file_name, 'w') as out:
        for i, doc_id in enumerate(ranked_list):
            next_doc_id = get_next_doc_id(doc_id)
            out.write(qrid + " Q0 " + next_doc_id + " 0 " + str(i + 1) + " pairs_seo\n")


def run_reranking(qrid, ranked_list, base_index, new_index, swig_path, scripts_dir, stopwords_file, queries_text_file,
                  jar_path, rank_model, output_dir, specific_ws_name='specific_ws',
                  new_feature_file_name='new_feature_file', feature_dir_name='feature_dir/',
                  new_trec_file_name='trec_file', score_file_name='score_file'):
    logger = logging.getLogger(sys.argv[0])
    ensure_dir(output_dir)
    specific_ws_path = output_dir + specific_ws_name
    feature_file_path = output_dir + new_feature_file_name
    score_file_path = output_dir + score_file_name
    trec_file_path = output_dir + new_trec_file_name
    full_feature_dir = output_dir + feature_dir_name

    create_reranking_ws(qrid, ranked_list, specific_ws_path)
    logger.info("creating features")
    features_file = create_features_file_diff(full_feature_dir, base_index, new_index, feature_file_path,
                                              specific_ws_path, scripts_dir, swig_path, stopwords_file,
                                              queries_text_file)
    logger.info("creating docname index")
    docname_index = create_index_to_doc_name_dict(features_file)
    logger.info("docname index creation is completed")
    query_index = create_index_to_query_dict(features_file)
    logger.info("features creation completed")
    logger.info("running ranking model on features file")
    run_model(features_file, jar_path, score_file_path, rank_model)
    logger.info("ranking completed")
    logger.info("retrieving scores")
    scores = retrieve_scores(docname_index, query_index, score_file_path)
    logger.info("scores retrieval completed")
    logger.info("creating trec_eval file")
    create_trec_eval_file(scores, trec_file_path)
    logger.info("trec file creation is completed")
    logger.info("ordering trec file")
    final = order_trec_file(trec_file_path)
    logger.info("ranking procedure completed")
    return final


@deprecated(reason='The functions this function uses have been altered')
def create_qrels(raw_ds, base_trec, out_file, ref, new_indices_dir, texts):
    # ind_name = {-1: "5", 1: "2"}
    # with open(out_file, 'w') as qrels:
    #     ranked_lists = read_raw_trec_file(base_trec)
    #     raw_stats = read_raw_ds(raw_ds)
    #
    #     ws_dir = "tmp_ws/"
    #     if not os.path.exists(ws_dir):
    #         os.makedirs(ws_dir)
    #     trectext_dir = "tmp_trectext/"
    #     if not os.path.exists(trectext_dir):
    #         os.makedirs(trectext_dir)
    #     trec_dir = "tmp_trec/"
    #     if not os.path.exists(trec_dir):
    #         os.makedirs(trec_dir)
    #     scores_dir = "tmp_scores/"
    #     if not os.path.exists(scores_dir):
    #         os.makedirs(scores_dir)
    #
    #     for qid in raw_stats:
    #         epoch, query = reverese_query(qid)
    #
    #         """ Change FOR GENERIC purposes if needed"""
    #         # if epoch not in ["04", "06"]:
    #         #     continue
    #
    #         for pair in raw_stats[qid]:
    #             ref_doc = pair.split("$")[0]
    #             out_index = int(pair.split("_")[1])
    #             query_write = query + epoch.lstrip('0') + ind_name[ref]
    #             name = generate_pair_name(pair)
    #             fname_pair = pair.replace("$", "_")
    #             feature_dir = "tmp_features/" + fname_pair + "/"
    #             if not os.path.exists(feature_dir):
    #                 os.makedirs(feature_dir)
    #             features_file = "qrels_features/" + fname_pair
    #             final_trec = run_reranking(ref_doc, feature_dir,,
    #             new_lists = read_raw_trec_file(final_trec)
    #             label = str(max(ranked_lists[qid].index(ref_doc) - new_lists[qid].index(ref_doc), 0))
    #             qrels.write(query_write + " 0 " + name + " " + label + "\n")
    raise Exception('This function is deprecated')


# TODO reconsider the use of index here
def create_bot_features(qrid, ref_index, top_docs_index, ranked_lists, doc_texts, output_dir, word_embed_model,
                        mode, base_index, new_index, queries_file, swig_path, doc_tfidf_dir, raw_ds_file,
                        documents_workingset_file, final_features_file, sentences_tfidf_dir='sentences_tfidf_dir/',
                        output_feature_files_dir='feature_files/', workingset_file='workingset.txt'):
    sentences_tfidf_dir = output_dir + sentences_tfidf_dir
    output_feature_files_dir = output_dir + output_feature_files_dir
    workingset_file = output_dir + workingset_file

    if mode == 'single':
        epoch, qid = reverese_query(qrid)

        create_raw_dataset(ranked_lists, doc_texts, raw_ds_file, ref_index, top_docs_index,
                           current_epoch=epoch, current_qid=qid)
        if is_file_empty(raw_ds_file):
            return True

        create_sentence_vector_files(sentences_tfidf_dir, raw_ds_file, base_index, new_index, swig_path,
                                     documents_workingset_file)
        query_text = get_query_text(queries_file, qid)
        feature_creation_single(raw_ds_file, ranked_lists, doc_texts, ref_index, top_docs_index,
                                doc_tfidf_dir, sentences_tfidf_dir, qrid, query_text, output_feature_files_dir,
                                final_features_file, workingset_file, word_embed_model)

    # elif mode == 'multiple':
    #     create_raw_dataset(ranked_lists, doc_texts, raw_ds_file, int(ref_index),
    #                        int(top_docs_index))
    #     create_sentence_vector_files(sentences_tfidf_dir, raw_ds_file, base_index, swig_path)
    #     queries = read_queries_file(queries_file)
    #     queries = transform_query_text(queries)
    #     # TODO update this function
    #     feature_creation_parallel(raw_ds_file, ranked_lists, doc_texts, int(top_docs_index),
    #                               int(ref_index), doc_tfidf_dir,
    #                               sentences_tfidf_dir, queries, output_feature_files_dir,
    #                               output_final_feature_file_dir, workingset_file)

    else:
        raise ValueError('mode value must be given, and it must be either \'single\' or \'multiple\'')

    return False
