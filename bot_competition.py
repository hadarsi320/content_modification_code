import logging
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from os.path import exists, basename, splitext

import numpy as np
from deprecated import deprecated
from lxml import etree
from nltk import sent_tokenize

from create_bot_features import update_text_doc
from gen_utils import run_and_print
from utils import get_qrid, create_trectext_file, parse_doc_id, \
    ensure_dirs, get_learning_data_path, get_doc_id
from vector_functionality import embedding_similarity, document_tfidf_similarity


def create_initial_trec_file(output_dir, qid, bots, only_bots, **kwargs):
    logger = logging.getLogger(sys.argv[0])

    new_trec_file = output_dir + 'trec_file_{}_{}'.format(qid, ','.join(bots))

    lines_written = 0
    ensure_dirs(new_trec_file)
    if 'trec_file' in kwargs:
        qrid = get_qrid(qid, 1)
        with open(kwargs['trec_file'], 'r') as trec_file:
            with open(new_trec_file, 'w') as new_file:
                for line in trec_file:
                    last_qrid = line.split()[0]
                    if last_qrid != qrid:
                        continue
                    pid = line.split()[2].split('-')[-1]
                    if not only_bots or pid in bots:
                        new_file.write(line)
                        lines_written += 1

    else:
        ranked_list = []
        with open(kwargs['positions_file'], 'r') as pos_file:
            for line in pos_file:
                doc_id = line.split()[2]
                epoch, last_qid, pid = parse_doc_id(doc_id)
                if epoch != '01' or last_qid != qid or (only_bots and pid not in bots):
                    continue
                if '_' in pid:
                    pid = pid.replace('_', '')
                position = int(line.split()[3])
                ranked_list.append([get_qrid(qid, 1), get_doc_id(1, qid, pid), 3 - position])
        ranked_list.sort(key=lambda x: x[2], reverse=True)
        with open(new_trec_file, 'w') as new_file:
            for file in ranked_list:
                new_file.write(f'{file[0]} Q0 {file[1]} 0 {file[2]} positions\n')
                lines_written += 1

    if lines_written == 0 and not only_bots:
        raise ValueError(f'query {qid} not in dataset')

    if only_bots and lines_written != len(bots):
        raise ValueError('Competitors {} not in dataset'.format(', '.join(kwargs['pid_list'])))

    logger.info('Competition trec file created')
    return new_trec_file


def create_initial_trectext_file(trectext_file, output_dir, qid, bots, only_bots):
    logger = logging.getLogger(sys.argv[0])

    new_trectext_file = output_dir + 'documents_{}_{}.trectext'.format(qid, ','.join(bots))
    ensure_dirs(new_trectext_file)

    parser = etree.XMLParser(recover=True)
    tree = ET.parse(trectext_file, parser=parser)
    root = tree.getroot()
    docs = {}
    for doc in root:
        pid = None
        for att in doc:
            if att.tag == 'DOCNO':
                doc_id = att.text
                epoch, last_qid, pid = parse_doc_id(doc_id)
                if epoch != '01' or last_qid != qid or (only_bots and pid not in bots):
                    break
                pid = pid.replace('_', '')
            elif att.tag == 'TEXT':
                docs[get_doc_id(1, qid, pid)] = '\n'.join(sent_tokenize(att.text))

    create_trectext_file(docs, new_trectext_file)
    logger.info('Competition trectext file created')
    return new_trectext_file


def generate_learning_dataset(output_dir, label_aggregation_method, seo_qrels, coherency_qrels, feature_fname):
    command = 'python dataset_creator.py ' + output_dir + ' ' + label_aggregation_method + ' ' + seo_qrels + ' ' + \
              coherency_qrels + ' ' + feature_fname
    run_and_print(command, 'Dataset Creator')


def create_model(svm_rank_scripts_dir, model_path, learning_data, svm_rank_c):
    ensure_dirs(model_path)
    command = f'{svm_rank_scripts_dir}svm_rank_learn -c {svm_rank_c} {learning_data} {model_path}'
    run_and_print(command, 'pair ranker learn')


def generate_predictions(model_path, svm_rank_scripts_dir, predictions_dir, feature_file):
    predictions_file = predictions_dir + '_predictions'.join(splitext(basename(feature_file)))
    ensure_dirs(predictions_file)
    command = f'{svm_rank_scripts_dir}svm_rank_classify {feature_file} {model_path} {predictions_file}'
    run_and_print(command, 'pair classify')
    return predictions_file


def get_highest_ranked_pair(features_file, predictions_file, threshold):
    """
    :param features_file: The features file, holds a line for every
    :param predictions_file: A file that holds a score for every line in the features file
    :return: (replacement_doc_id, out_index, in_index)
    """
    with open(features_file, 'r') as f:
        pairs = [line.rstrip('\n').split('# ')[1] for line in f if len(line) > 0]

    with open(predictions_file, 'r') as f:
        scores = [float(line) for line in f if len(line) > 0]

    max_pair, score = max(zip(pairs, scores), key=lambda x: x[1])

    if threshold is not None and score < threshold:
        return None

    rep_doc_id, out_index, in_index = max_pair.split('_')
    return rep_doc_id, int(out_index), int(in_index)


@deprecated(reason='This version uses the sentences from the raw dataset file, which are cleaned and shouldnt be used')
def generate_updated_document_dep(max_pair, raw_ds_file, doc_texts):
    with open(raw_ds_file) as f:
        for line in f:
            pair = line.split('\t')[1]
            key = pair.split('$')[1]
            if key == max_pair:
                ref_doc_id = pair.split('$')[0]
                sentence_in = line.split('\t')[3].strip('\n')
                sentence_out_index = int(key.split('_')[1])
                break

    ref_doc = doc_texts[ref_doc_id]
    return update_text_doc(ref_doc, sentence_in, sentence_out_index)


def generate_updated_document(doc_texts, ref_doc_id, rep_doc_id, out_index, in_index):
    ref_doc = sent_tokenize(doc_texts[ref_doc_id])
    rep_doc = sent_tokenize(doc_texts[rep_doc_id])
    ref_doc[out_index] = rep_doc[in_index]
    return '\n'.join(ref_doc)


def get_ranked_competitors_list(trec_file, current_epoch):
    competitors_ranked_list = []
    with open(trec_file, 'r') as f:
        for line in f:
            epoch = int(line.split()[0][-2:])
            if epoch != current_epoch:
                continue
            doc_id = line.split()[2]
            competitors_ranked_list.append(parse_doc_id(doc_id)[2])
    return competitors_ranked_list


def advance_round(line):
    split_line = line.split()

    qrid = split_line[0]
    qid = qrid[:-2]
    epoch = int(qrid[-2:]) + 1
    split_line[0] = qid + str(epoch).zfill(2)

    doc_id = split_line[2].split('-')
    doc_id[1] = str(epoch).zfill(2)
    split_line[2] = '-'.join(doc_id)
    return ' '.join(split_line)


def update_trec_file(comp_trec_file, reranked_trec_file):
    with open(comp_trec_file, 'a') as trec:
        with open(reranked_trec_file, 'r') as reranked_trec:
            for line in reranked_trec:
                trec.write(line)


def generate_document_tfidf_files(workingset_file, output_dir, swig_path, base_index, new_index):
    ensure_dirs(output_dir)
    command = f'java -Djava.library.path={swig_path} -cp seo_indri_utils.jar PrepareTFIDFVectorsWSDiff ' \
              f'{base_index} {new_index} {workingset_file} {output_dir}'
    run_and_print(command, command_name='Document tfidf Creation')


def record_doc_similarity(doc_texts, current_epoch, similarity_file, word_embedding_model, document_tfidf_dir):
    logger = logging.getLogger(sys.argv[0])
    ensure_dirs(similarity_file)

    recent_documents = []
    recent_texts = []
    for document in doc_texts:
        epoch = int(document.split('-')[1])
        if epoch == current_epoch:
            recent_documents.append(document)
            recent_texts.append(doc_texts[document])
    assert len(recent_documents) == 2

    tfidf_sim = document_tfidf_similarity(*[document_tfidf_dir + doc for doc in recent_documents])
    embedding_sim = embedding_similarity(*recent_texts, word_embedding_model)
    with open(similarity_file, 'a') as f:
        if current_epoch == 1:
            f.write('Round\ttfidf\tembedding\n')
        f.write(f'{current_epoch - 1}\t{round(tfidf_sim, 3)}\t{round(embedding_sim, 3)}\n')
    logger.info('Recorded document similarity')


def record_replacement(replacements_file, epoch, in_doc_id, out_doc_id, out_index, in_index):
    ensure_dirs(replacements_file)
    with open(replacements_file, 'a') as f:
        f.write(f'{epoch}. {in_doc_id}\t{out_doc_id}\t{out_index}\t{in_index}\n')


def create_pair_ranker(model_path, label_aggregation_method, label_aggregation_b, svm_rank_c,
                       aggregated_data_dir, seo_qrels_file, coherency_qrels_file, unranked_features_file,
                       svm_rank_scripts_dir):
    learning_data_dir = aggregated_data_dir + 'feature_sets/'
    learning_data_path = get_learning_data_path(learning_data_dir, label_aggregation_method, label_aggregation_b)

    if not exists(learning_data_path):
        generate_learning_dataset(aggregated_data_dir, label_aggregation_method,
                                  seo_qrels_file, coherency_qrels_file,
                                  unranked_features_file)
    create_model(svm_rank_scripts_dir, model_path, learning_data_path, svm_rank_c)


def get_rankings(trec_file, bot_ids, qid, epoch):
    """
    :param trec_file: a trecfile
    :param bot_ids: the pids of the players who are bots
    :param qid: query id
    :param epoch: current round
    :return: two dictionaries of the form {pid: location}, one for the bots and the other for the students
    """

    bots = {}
    students = {}
    position = 0
    epoch = str(epoch).zfill(2)
    with open(trec_file, 'r') as f:
        for line in f:
            doc_id = line.split()[2]
            last_epoch, last_qid, pid = parse_doc_id(doc_id)
            if last_epoch != epoch or last_qid != qid:
                continue
            if pid in bot_ids:
                bots[pid] = position
            else:
                students[pid] = position
            position += 1
    return bots, students


def find_fastest_climbing_document(ranked_list, qid, past=1):
    if len(ranked_list) <= past:
        return None

    past_rank_change = defaultdict(list)
    pid_list = [parse_doc_id(doc_id)[2] for doc_id in next(iter(ranked_list.values()))[qid]]

    for pid in pid_list:
        last_rank = None
        for epoch in sorted(ranked_list)[-(past + 1):]:
            rank = ranked_list[epoch][qid].index(get_doc_id(epoch, qid, pid))
            if last_rank is not None:
                past_rank_change[pid].append(last_rank - rank)
            last_rank = rank

    average_rank_change = {pid: np.average(past_rank_change[pid]) for pid in pid_list}
    ordered_rising_documents = sorted(past_rank_change, key=lambda x: average_rank_change[x], reverse=True)

    last_epoch = sorted(ranked_list)[-1]
    fastest_rising_doc = ordered_rising_documents[0]
    if ranked_list[last_epoch][qid].index(get_doc_id(last_epoch, qid, fastest_rising_doc)) == 0:
        # we do not want to return the document that is ranked first as that is us
        fastest_rising_doc = ordered_rising_documents[1]

    if average_rank_change[fastest_rising_doc] > 0:
        return fastest_rising_doc
    else:
        return None


def get_last_top_document(ranked_list, qid):
    if len(ranked_list) == 1:
        return None

    last_round, current_round = sorted(ranked_list)[-2:]

    last_top_doc_id = ranked_list[last_round][qid][0]
    current_top_doc_id = ranked_list[current_round][qid][0]

    if parse_doc_id(last_top_doc_id)[2] == parse_doc_id(current_top_doc_id)[2]:
        return None

    return last_top_doc_id


def get_target_documents(top_refinement, qid, epoch, ranked_lists, past_targets):
    if top_refinement == 'acceleration':
        fastest_rising = find_fastest_climbing_document(ranked_lists, qid)
        target_documents = [get_doc_id(epoch, qid, fastest_rising)] if fastest_rising is not None \
            else None

    elif top_refinement == 'past_top':
        past_top = get_last_top_document(ranked_lists, qid)
        target_documents = [past_top] if past_top is not None else None

    elif top_refinement == 'highest_rated_inferiors':
        target_documents = ranked_lists[str(epoch).zfill(2)][qid][1:3]

    elif top_refinement == 'past_targets' and qid in past_targets:
        target_documents = past_targets[qid]

    elif top_refinement == 'everything':
        target_documents = []
        for method in ['acceleration', 'past_top', 'highest_rated_inferiors', 'past_targets']:
            targets = get_target_documents(method, qid, epoch, ranked_lists, past_targets)
            if targets is None:
                continue
            for target in targets:
                if target not in target_documents:
                    target_documents.append(target)

    else:
        target_documents = None

    return target_documents
