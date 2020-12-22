import logging
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
from nltk import sent_tokenize

import constants
import utils
from gen_utils import run_and_print
from readers import TrecReader
from utils import clean_texts, get_java_object, create_trectext_file, run_model, create_features_file_diff, \
    create_trec_eval_file, order_trec_file, retrieve_scores, get_query_text, parse_qrid, create_index_to_query_dict,\
    generate_pair_name, ensure_dirs, tokenize_document, is_file_empty, get_next_doc_id, get_next_qrid
from vector_functionality import query_term_freq, embedding_similarity, similarity_to_centroid_tf_idf, \
    document_centroid, similarity_to_centroid_semantic, get_text_centroid, add_dict, cosine_similarity


def create_sentence_pairs(top_docs, ref_doc, texts):
    result = {}
    ref_sentences = tokenize_document(texts[ref_doc])
    for doc in top_docs:
        doc_sentences = tokenize_document(texts[doc])
        for in_index, top_sentence in enumerate(doc_sentences):
            if top_sentence in ref_sentences or top_sentence == '':
                continue
            for out_index, ref_sentence in enumerate(ref_sentences):
                key = ref_doc + "$" + doc + "_" + str(out_index) + "_" + str(in_index)
                result[key] = ref_sentence + "\t" + top_sentence
    return result


def create_raw_dataset(ranked_lists, doc_texts, output_file, ref_index, copy_docs, **kwargs):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as output:
        for epoch in ranked_lists.get_epochs():
            if 'epoch' in kwargs and epoch != kwargs['epoch']:
                continue
            for qid in ranked_lists[epoch]:
                if 'qid' in kwargs and qid != kwargs['qid']:
                    continue

                ref_doc = ranked_lists[epoch][qid][ref_index]
                pairs = create_sentence_pairs(copy_docs, ref_doc, doc_texts)
                for key in pairs:
                    output.write(qid.lstrip('0') + str(epoch) + '\t' + key + '\t' + pairs[key] + '\n')


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


def write_files(feature_list, feature_vals, output_dir, qrid, ref_index):
    epoch, qid = parse_qrid(qrid)
    query_write = f'{qid}{epoch.lstrip("0")}{ref_index + 1}'

    for feature in feature_list:
        with open(output_dir + "doc" + feature + "_" + query_write, 'w') as out:
            for pair in feature_vals[feature]:
                name = generate_pair_name(pair)
                value = feature_vals[feature][pair]
                out.write(name + " " + str(value) + "\n")


def create_features(qrid, ranked_lists, doc_texts, ref_index, target_docs, doc_tfidf_vectors_dir,
                    sentence_tfidf_vectors_dir, query_text, output_dir, raw_ds, word_embed_model):
    feature_vals = defaultdict(dict)
    relevant_pairs = raw_ds[qrid]
    epoch, qid = parse_qrid(qrid)
    query_text = clean_texts(query_text)

    feature_list = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                    "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                    "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                    "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef"]
    #                 "SimilarityToPredRef"]

    past_winners = get_past_winners(ranked_lists, epoch, qid)
    past_winners_semantic_centroid_vector = past_winners_centroid(past_winners, doc_texts, word_embed_model, True)
    past_winners_tfidf_centroid_vector = get_past_winners_tfidf_centroid(past_winners, doc_tfidf_vectors_dir)

    top_doc_upgrade = ref_index == 0
    ref_doc = ranked_lists[epoch][qid][ref_index]
    ref_sentences = sent_tokenize(doc_texts[ref_doc])
    # top_docs_tfidf_centroid = document_centroid([get_java_object(doc_tfidf_vectors_dir + doc) for doc in target_docs])
    top_docs_tfidf_centroid = document_centroid(doc_tfidf_vectors_dir, target_docs)
    for pair in relevant_pairs:
        sentence_in = relevant_pairs[pair]["in"]
        sentence_out = relevant_pairs[pair]["out"]
        in_vec = get_text_centroid(sentence_in, word_embed_model, True)
        out_vec = get_text_centroid(sentence_out, word_embed_model, True)
        replace_index = int(pair.split("_")[1])

        # Query features
        feature_vals['FractionOfQueryWordsIn'][pair] = query_term_freq("avg", sentence_in, query_text)
        feature_vals['FractionOfQueryWordsOut'][pair] = query_term_freq("avg", sentence_out, query_text)

        # Target documents features
        if not top_doc_upgrade:
            feature_vals['CosineToCentroidIn'][pair] = similarity_to_centroid_tf_idf(
                sentence_tfidf_vectors_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2],
                top_docs_tfidf_centroid)
            feature_vals['CosineToCentroidOut'][pair] = similarity_to_centroid_tf_idf(
                sentence_tfidf_vectors_dir + pair.split("$")[0] + "_" + pair.split("_")[1],
                top_docs_tfidf_centroid)
            feature_vals["CosineToCentroidInVec"][pair] = \
                similarity_to_centroid_semantic(sentence_in, doc_texts, target_docs, word_embed_model, True)
            feature_vals["CosineToCentroidOutVec"][pair] = \
                similarity_to_centroid_semantic(sentence_out, doc_texts, target_docs, word_embed_model, True)
        else:
            feature_vals['CosineToCentroidIn'][pair] = 0
            feature_vals['CosineToCentroidOut'][pair] = 0
            feature_vals["CosineToCentroidInVec"][pair] = 0
            feature_vals["CosineToCentroidOutVec"][pair] = 0

        # Top documents focused features
        feature_vals['CosineToWinnerCentroidInVec'][pair] = \
            cosine_similarity(in_vec, past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidOutVec'][pair] = \
            cosine_similarity(out_vec, past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidIn'][pair] = similarity_to_centroid_tf_idf(
            sentence_tfidf_vectors_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2],
            past_winners_tfidf_centroid_vector)
        feature_vals['CosineToWinnerCentroidOut'][pair] = similarity_to_centroid_tf_idf(
            sentence_tfidf_vectors_dir + pair.split("$")[0] + "_" + pair.split("_")[1],
            past_winners_tfidf_centroid_vector)

        # Readability features
        feature_vals['SimilarityToPrev'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_in, "prev", word_embed_model, True)
        feature_vals['SimilarityToRefSentence'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_in, "own", word_embed_model, True)
        feature_vals['SimilarityToPred'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_in, "pred", word_embed_model, True)
        feature_vals['SimilarityToPrevRef'][pair] = \
            context_similarity(replace_index, ref_sentences, sentence_out, "prev", word_embed_model, True)
    write_files(feature_list, feature_vals, output_dir, qrid, ref_index)


def create_features_og(raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index, doc_tfidf_vectors_dir,
                       tfidf_sentence_dir, queries, output_dir, qrid):
    global word_embd_model
    feature_vals = defaultdict(dict)
    relevant_pairs = raw_ds[qrid]
    epoch, qid = parse_qrid(qrid)
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
    # top_docs_tfidf_centroid = document_centroid([get_java_object(doc_tfidf_vectors_dir + doc) for doc in top_docs])
    top_docs_tfidf_centroid = document_centroid(doc_tfidf_vectors_dir, top_docs)
    for pair in relevant_pairs:
        sentence_in = relevant_pairs[pair]["in"]
        sentence_out = relevant_pairs[pair]["out"]
        in_vec = get_text_centroid(clean_texts(sentence_in), word_embd_model, True)
        out_vec = get_text_centroid(clean_texts(sentence_out), word_embd_model, True)
        replace_index = int(pair.split("_")[1])

        feature_vals['FractionOfQueryWordsIn'][pair] = query_term_freq("avg", sentence_in, query_text)
        feature_vals['FractionOfQueryWordsOut'][pair] = query_term_freq("avg", sentence_out, query_text)
        feature_vals['CosineToCentroidIn'][pair] = similarity_to_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2], top_docs_tfidf_centroid)
        feature_vals['CosineToCentroidOut'][pair] = similarity_to_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[0] + "_" + pair.split("_")[1], top_docs_tfidf_centroid)
        feature_vals["CosineToCentroidInVec"][pair] = similarity_to_centroid_semantic(sentence_in, doc_texts, top_docs,
                                                                                      word_embd_model, True)
        feature_vals["CosineToCentroidOutVec"][pair] = similarity_to_centroid_semantic(sentence_out, doc_texts,
                                                                                       top_docs, word_embd_model, True)
        feature_vals['CosineToWinnerCentroidInVec'][pair] = cosine_similarity(in_vec,
                                                                              past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidOutVec'][pair] = cosine_similarity(out_vec,
                                                                               past_winners_semantic_centroid_vector)
        feature_vals['CosineToWinnerCentroidIn'][pair] = similarity_to_centroid_tf_idf(
            tfidf_sentence_dir + pair.split("$")[1].split("_")[0] + "_" + pair.split("_")[2],
            past_winners_tfidf_centroid_vector)
        feature_vals['CosineToWinnerCentroidOut'][pair] = similarity_to_centroid_tf_idf(
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


# def feature_creation_parallel(raw_dataset_file, ranked_lists, doc_texts, top_doc_index, ref_doc_index,
#                               doc_tfidf_vectors_dir, tfidf_sentence_dir, queries, output_feature_files_dir,
#                               output_final_features_dir, workingset_file):
#     global word_embd_model
#     args = [qid for qid in queries]
#     if not os.path.exists(output_feature_files_dir):
#         os.makedirs(output_feature_files_dir)
#     if not os.path.exists(output_final_features_dir):
#         os.makedirs(output_final_features_dir)
#     raw_ds = read_raw_ds(raw_dataset_file)
#     create_ws(raw_ds, workingset_file, ref_doc_index)
#     func = partial(create_features_og, raw_ds, ranked_lists, doc_texts, top_doc_index, ref_doc_index,
#                    doc_tfidf_vectors_dir, tfidf_sentence_dir, queries, output_feature_files_dir)
#     workers = cpu_count() - 1
#     list_multiprocessing(args, func, workers=workers)
#     command = "perl scripts/generateSentences.pl " + output_feature_files_dir + " " + workingset_file
#     run_bash_command(command)
#     run_bash_command("mv features " + output_final_features_dir)


def feature_creation(qrid, ranked_lists, doc_texts, ref_index, copy_docs, doc_tfidf_vectors_dir,
                     sentence_tfidf_vectors_dir, raw_dataset_file, query_text, output_feature_files_dir,
                     output_final_features_file, workingset_file, word_embed_model):
    ensure_dirs(output_feature_files_dir, output_final_features_file)
    raw_ds = read_raw_ds(raw_dataset_file)
    create_ws(raw_ds, workingset_file, ref_index)
    create_features(qrid, ranked_lists, doc_texts, ref_index, copy_docs, doc_tfidf_vectors_dir,
                    sentence_tfidf_vectors_dir, query_text, output_feature_files_dir, raw_ds, word_embed_model)

    constants.lock.acquire()
    command = f"perl scripts/generateSentences.pl {output_feature_files_dir} {workingset_file}"
    run_and_print(command, 'generateSentences.pl')
    command = "mv features " + output_final_features_file
    run_and_print(command, 'move')
    constants.lock.release()


def run_svm_rank_model(test_file, model_file, predictions_folder):
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    predictions_file = predictions_folder + os.path.basename(model_file)
    command = "./svm_rank_classify " + test_file + " " + model_file + " " + predictions_file
    run_and_print(command, command_name='pair ranking')
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


def create_ws(raw_ds, ws_fname, ref_index):
    ensure_dirs(ws_fname)
    with open(ws_fname, 'w') as ws:
        for qrid in raw_ds:
            epoch, qid = parse_qrid(qrid)
            query_write = f'{qid}{epoch.lstrip("0")}{ref_index + 1}'
            for i, pair in enumerate(raw_ds[qrid]):
                name = generate_pair_name(pair)
                ws.write(query_write + " Q0 " + name + " 0 " + str(i + 1) + " pairs_seo\n")


def create_new_trectext(doc, texts, new_text, new_trectext_name):
    text_copy = deepcopy(texts)
    text_copy[doc] = new_text
    create_trectext_file(text_copy, new_trectext_name, "ws_debug")


def create_reranking_ws(qrid, raw_ranked_lists, file_name):
    next_qrid = get_next_qrid(qrid)
    with open(file_name, 'w') as out:
        for i, doc_id in enumerate(raw_ranked_lists[qrid]):
            next_doc_id = get_next_doc_id(doc_id)
            out.write(next_qrid + " Q0 " + next_doc_id + " 0 " + str(i + 1) + " pairs_seo\n")


def run_reranking(qrid, trec_file, base_index, new_index, swig_path, scripts_dir, stopwords_file, queries_text_file,
                  jar_path, rank_model, output_dir, reranking_ws_name='reranking_ws',
                  new_feature_file_name='new_feature_file', feature_dir_name='feature_dir/',
                  new_trec_file_name='trec_file', score_file_name='score_file'):
    logger = logging.getLogger(sys.argv[0])
    ensure_dirs(output_dir)

    reranked_trec_file = output_dir + new_trec_file_name
    feature_file = output_dir + new_feature_file_name
    full_feature_dir = output_dir + feature_dir_name
    reranking_ws = output_dir + reranking_ws_name
    score_file = output_dir + score_file_name

    raw_ranked_lists = TrecReader(trec_file, raw=True)
    create_reranking_ws(qrid, raw_ranked_lists, reranking_ws)
    logger.info("creating features")
    features_file = create_features_file_diff(full_feature_dir, base_index, new_index, feature_file, reranking_ws,
                                              scripts_dir, swig_path, stopwords_file, queries_text_file)
    logger.info("creating docname index")
    docname_index = create_index_to_doc_name_dict(features_file)
    logger.info("docname index creation is completed")
    query_index = create_index_to_query_dict(features_file)
    logger.info("features creation completed")
    logger.info("running ranking model on features file")
    run_model(features_file, jar_path, score_file, rank_model)
    logger.info("ranking completed")
    logger.info("retrieving scores")
    scores = retrieve_scores(docname_index, query_index, score_file)
    logger.info("scores retrieval completed")
    logger.info("creating trec_eval file")
    create_trec_eval_file(scores, reranked_trec_file)
    logger.info("trec file creation is completed")
    logger.info("ordering trec file")
    final = order_trec_file(reranked_trec_file)
    logger.info("ranking procedure completed")
    return final


def create_bot_features(qrid, ref_index, ranked_lists, doc_texts, target_docs, output_dir, word_embed_model,
                        base_index, new_index, queries_file, swig_path, doc_tfidf_dir, raw_ds_file,
                        documents_workingset_file, final_features_file, sentences_tfidf_dir='sentences_tfidf_dir/',
                        output_feature_files_dir='feature_files/', workingset_file='workingset.txt'):
    sentences_tfidf_dir = output_dir + sentences_tfidf_dir
    output_feature_files_dir = output_dir + output_feature_files_dir
    workingset_file = output_dir + workingset_file

    epoch, qid = parse_qrid(qrid)
    create_raw_dataset(ranked_lists, doc_texts, raw_ds_file, ref_index, target_docs, epoch=epoch, qid=qid)
    if is_file_empty(raw_ds_file):
        return True

    create_sentence_vector_files(sentences_tfidf_dir, raw_ds_file, base_index, new_index, swig_path,
                                 documents_workingset_file)
    query_text = get_query_text(queries_file, qid)
    feature_creation(qrid, ranked_lists, doc_texts, ref_index, target_docs, doc_tfidf_dir, sentences_tfidf_dir,
                     raw_ds_file, query_text, output_feature_files_dir, final_features_file, workingset_file,
                     word_embed_model)

    return False
