import logging
import os
import sys
import xml.etree.ElementTree as ET
from os.path import exists, basename, splitext

from deprecated import deprecated
from lxml import etree
from nltk import sent_tokenize

from create_bot_features import update_text_doc
from gen_utils import run_and_print
from utils import get_qrid, create_trectext_file, parse_doc_id, \
    ensure_dir, create_documents_workingset, get_learning_data_path, get_doc_id
from vector_functionality import centroid_similarity, document_tfidf_similarity


def create_initial_trec_file(output_dir, qid, trec_file=None, positions_file=None, pid_list=None,
                             dummy_bot_index=None):
    assert bool(positions_file) != bool(trec_file) and bool(pid_list) != bool(dummy_bot_index) and \
           bool(trec_file) == bool(pid_list)
    logger = logging.getLogger(sys.argv[0])

    new_trec_file = output_dir + f'trec_file_{qid}_'
    if pid_list:
        new_trec_file += ','.join(pid_list)
    else:
        new_trec_file += str(dummy_bot_index)

    ensure_dir(new_trec_file)
    qrid = get_qrid(qid, 1)
    if trec_file:
        lines_written = 0
        with open(trec_file, 'r') as trec_file:
            with open(new_trec_file, 'w') as new_file:
                for line in trec_file:
                    last_qrid = line.split()[0]
                    if last_qrid != qrid:
                        continue
                    pid = line.split()[2].split('-')[-1]
                    if pid_list is None or pid in pid_list:
                        new_file.write(line)
                        lines_written += 1
        if pid_list and lines_written != len(pid_list):
            raise ValueError('Competitors/Qid not in dataset')

    else:
        ranked_list = []
        with open(positions_file, 'r') as pos_file:
            for line in pos_file:
                doc_id = line.split()[2]
                epoch, last_qid, pid = parse_doc_id(doc_id)
                if epoch != '01' or last_qid != qid or (pid_list and pid not in pid_list):
                    continue
                if '_' in pid:
                    pid = pid.replace('_', '')
                position = int(line.split()[3])
                ranked_list.append([get_qrid(qid, 1), get_doc_id(1, qid, pid), 3 - position])
        ranked_list.sort(key=lambda x: x[2], reverse=True)
        with open(new_trec_file, 'w') as new_file:
            for file in ranked_list:
                new_file.write(f'{file[0]} Q0 {file[1]} 0 {file[2]} positions\n')

    logger.info('Competition trec file created')
    return new_trec_file


def create_initial_trectext_file(full_trectext_file, output_dir, qid, pid_list=None, dummy_bot_index=None):
    assert bool(pid_list) != bool(dummy_bot_index)
    logger = logging.getLogger(sys.argv[0])

    if pid_list:
        new_trectext_file = output_dir + f'documents_{qid}_{",".join(pid_list)}.trectext'
    else:
        new_trectext_file = output_dir + f'documents_{qid}_{dummy_bot_index}.trectext'
    ensure_dir(new_trectext_file)

    parser = etree.XMLParser(recover=True)
    tree = ET.parse(full_trectext_file, parser=parser)
    root = tree.getroot()
    docs = {}
    for doc in root:
        pid = None
        for att in doc:
            if att.tag == 'DOCNO':
                doc_id = att.text
                epoch, last_qid, pid = parse_doc_id(doc_id)
                if epoch != '01' or last_qid != qid or (pid_list and pid not in pid_list):
                    break
                if '_' in pid:
                    pid = pid.replace('_', '')
            elif att.tag == 'TEXT':
                docs[get_doc_id(1, qid, pid)] = '\n'.join(sent_tokenize(att.text))

    create_trectext_file(docs, new_trectext_file)
    logger.info('Competition trectext file created')
    return new_trectext_file


@deprecated(reason='The module create_bot_features no longer has a main')
def create_features(qrid, trec_file, trectext_file, raw_ds_file, doc_tdidf_dir, index, output_dir,
                    mode='single', ref_index=1, top_docs_index=1):
    command = f'python create_bot_features.py --mode={mode} --qrid={qrid} --ref_index={ref_index} ' \
              f'--top_docs_index={top_docs_index} --trec_file={trec_file} --trectext_file={trectext_file} ' \
              f'--raw_ds_out={raw_ds_file} --doc_tfidf_dir={doc_tdidf_dir} --index_path={index} ' \
              f'--output_dir={output_dir}'
    run_and_print(command)


def generate_learning_dataset(output_dir, label_aggregation_method, seo_qrels, coherency_qrels, feature_fname):
    command = 'python dataset_creator.py ' + output_dir + ' ' + label_aggregation_method + ' ' + seo_qrels + ' ' + \
              coherency_qrels + ' ' + feature_fname
    run_and_print(command)


def create_model(svm_rank_scripts_dir, model_path, learning_data, svm_rank_c):
    ensure_dir(model_path)
    command = svm_rank_scripts_dir + 'svm_rank_learn -c ' + svm_rank_c + ' ' + learning_data + ' ' + model_path
    run_and_print(command)


def generate_predictions(model_path, svm_rank_scripts_dir, predictions_dir, feature_file):
    predictions_file = predictions_dir + '_predictions'.join(splitext(basename(feature_file)))
    ensure_dir(predictions_file)
    command = svm_rank_scripts_dir + 'svm_rank_classify ' + feature_file + ' ' + model_path + ' ' + predictions_file
    run_and_print(command)
    return predictions_file


def get_highest_ranked_pair(features_file, predictions_file):
    """
    :param features_file: The features file, holds a line for every
    :param predictions_file: A file that holds a score for every line in the features file
    :return: (replacement_doc_id, out_index, in_index)
    """
    with open(features_file, 'r') as f:
        pairs = [line.rstrip('\n').split('# ')[1] for line in f if len(line) > 0]

    with open(predictions_file, 'r') as f:
        scores = [float(line) for line in f if len(line) > 0]

    max_pair, _ = max(zip(pairs, scores), key=lambda x: x[1])
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
    # TODO use update texts for the multiple competitions version
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


@deprecated(reason='Created for dumb reason')
def get_doc_text(doctext_file, doc_id):
    xml_parser = etree.XMLParser(recover=True)
    tree = ET.parse(doctext_file, parser=xml_parser)
    root = tree.getroot()
    last_doc_id = ''
    for doc in root:
        for att in doc:
            if att.tag == 'DOCNO':
                last_doc_id = att.text
            elif att.tag == 'TEXT' and (last_doc_id == doc_id):
                return att.text
    raise ValueError('No document was found in path {} with the doc_id {}'
                     .format(doctext_file, doc_id))


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


def append_to_trec_file(comp_trec_file, reranked_trec_file):
    with open(comp_trec_file, 'a') as trec:
        with open(reranked_trec_file, 'r') as reranked_trec:
            for line in reranked_trec:
                trec.write(advance_round(line) + '\n')


def generate_document_tfidf_files(workingset_file, output_dir, swig_path, base_index, new_index):
    ensure_dir(output_dir)
    command = f'java -Djava.library.path={swig_path} -cp seo_indri_utils.jar PrepareTFIDFVectorsWSDiff ' \
              f'{base_index} {new_index} {workingset_file} {output_dir}'
    run_and_print(command, command_name='Document tfidf Creation')


def record_doc_similarity(doc_texts, current_epoch, similarity_file, word_embedding_model, document_tfidf_dir):
    logger = logging.getLogger(sys.argv[0])
    ensure_dir(similarity_file)

    recent_documents = []
    recent_texts = []
    for document in doc_texts:
        epoch = int(document.split('-')[1])
        if epoch == current_epoch:
            recent_documents.append(document)
            recent_texts.append(doc_texts[document])
    assert len(recent_documents) == 2

    tfidf_similarity = document_tfidf_similarity(*[document_tfidf_dir + doc for doc in recent_documents])
    embedding_similarity = centroid_similarity(*recent_texts, word_embedding_model)
    with open(similarity_file, 'a') as f:
        if current_epoch == 1:
            f.write('Round\ttfidf\tembedding\n')
        f.write(f'{current_epoch - 1}\t{round(tfidf_similarity, 3)}\t{round(embedding_similarity, 3)}\n')
    logger.info('Recorded document similarity')


def record_replacement(replacements_file, epoch, in_doc_id, out_doc_id, out_index, in_index):
    ensure_dir(replacements_file)
    with open(replacements_file, 'a') as f:
        f.write(f'{epoch}. {in_doc_id}\t{out_doc_id}\t{out_index}\t{in_index}\n')


def create_pair_ranker(model_path, label_aggregation_method, label_aggregation_b, svm_rank_c,
                       aggregated_data_dir, seo_qrels_file, coherency_qrels_file, unranked_features_file,
                       svm_rank_scripts_dir):
    if not exists(model_path):
        learning_data_dir = aggregated_data_dir + 'feature_sets/'
        learning_data_path = get_learning_data_path(learning_data_dir, label_aggregation_method, label_aggregation_b)

        if not exists(learning_data_path):
            generate_learning_dataset(aggregated_data_dir, label_aggregation_method,
                                      seo_qrels_file, coherency_qrels_file,
                                      unranked_features_file)
        create_model(svm_rank_scripts_dir, model_path, learning_data_path, svm_rank_c)


def get_rankings(trec_file, dummy_bot_index, qid, epoch):
    """
    :param trec_file: a trecfile
    :param dummy_bot_index: the index of the dummy who's a bot
    :param qid: query id
    :param epoch: current round
    :return: two dictionaries of the form {pid: location}, one for the bots and the other for the students
    """
    assert int(dummy_bot_index) in [1, 2]
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
            if pid in ['BOT', f'DUMMY{dummy_bot_index}']:
                bots[pid] = position
            else:
                students[pid] = position
            position += 1
    return bots, students


def get_competitors(positions_file, qid):
    competitors_list = []
    with open(positions_file, 'r') as f:
        for line in f:
            doc_id = line.split()[2]
            _, last_qid, pid = parse_doc_id(doc_id)
            pid = pid.replace('_', '')
            if last_qid == qid and pid not in competitors_list:
                competitors_list.append(pid)
    return competitors_list
