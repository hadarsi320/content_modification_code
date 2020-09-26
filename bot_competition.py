import os
import xml.etree.ElementTree as ET
from os.path import exists, basename, splitext

from deprecated import deprecated
from lxml import etree
from nltk import sent_tokenize

from create_bot_features import update_text_doc
from gen_utils import run_and_print
from utils import get_qrid, create_trectext_file, parse_doc_id, \
    generate_doc_id, ensure_dir
from vector_functionality import centroid_similarity


def create_initial_trec_file(original_trec_file: str, output_dir: str, qid: str, competitors: list):
    new_trec_file = output_dir + 'trec_file_' + qid + '_' + ','.join(competitors)
    ensure_dir(new_trec_file)
    qrid = get_qrid(qid, 1)
    lines_written = 0
    with open(original_trec_file, 'r') as trec_file:
        with open(new_trec_file, 'w') as f:
            for line in trec_file:
                line_qrid = line.split()[0]
                if int(line_qrid) > int(qrid):
                    break
                competitor = line.split()[2].split('-')[-1]
                if qrid == line_qrid and competitor in competitors:
                    f.write(line)
                    lines_written += 1
    if lines_written != len(competitors):
        raise ValueError('Competitors/Qid not in dataset')
    return new_trec_file


def create_initial_trectext_file(original_trectext_file, output_dir, qid, competitors):
    # TODO trim and preprocess the documents
    if not exists(output_dir):
        os.makedirs(output_dir)
    doc_id_list = [generate_doc_id(1, qid, competitor) for competitor in competitors]
    new_trectext_file = output_dir + f'documents_{qid}_{",".join(competitors)}.trectext'

    parser = etree.XMLParser(recover=True)
    tree = ET.parse(original_trectext_file, parser=parser)
    root = tree.getroot()
    docs = {}
    for doc in root:
        name = ""
        for att in doc:
            if att.tag == 'DOCNO':
                name = att.text
                if name not in doc_id_list:
                    break
            elif att.tag == 'TEXT':
                docs[name] = '\n'.join(sent_tokenize(att.text))

    create_trectext_file(docs, new_trectext_file)
    return new_trectext_file


@deprecated(reason='The module create_bot_features no longer has a main')
def create_features(qrid, trec_file, trectext_file, raw_ds_file, doc_tdidf_dir, index, output_dir,
                    mode='single', ref_index=1, top_docs_index=1):
    # TODO replace this with a function
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


def generate_predictions(model_path, svm_rank_scripts_dir, output_dir, feature_file):
    rankings_dir = output_dir + 'predictions/'
    if not exists(rankings_dir):
        os.makedirs(rankings_dir)
    rankings_file = rankings_dir + '_predictions'.join(splitext(basename(feature_file)))
    command = svm_rank_scripts_dir + 'svm_rank_classify ' + feature_file + ' ' + model_path + ' ' + rankings_file
    run_and_print(command)
    return rankings_file


def get_highest_ranked_pair(features_file, predictions_file):
    with open(features_file, 'r') as f:
        pairs = [line.rstrip('\n').split('# ')[1] for line in f if len(line) > 0]

    with open(predictions_file, 'r') as f:
        scores = [float(line) for line in f if len(line) > 0]

    max_pair, _ = max(zip(pairs, scores), key=lambda x: x[1])
    return max_pair


# def generate_updated_document(max_pair, raw_ds_file, doc_texts):
#     # TODO use update texts for the multiple competitions version
#     with open(raw_ds_file) as f:
#         for line in f:
#             pair = line.split('\t')[1]
#             key = pair.split('$')[1]
#             if key == max_pair:
#                 ref_doc_id = pair.split('$')[0]
#                 sentence_in = line.split('\t')[3].strip('\n')
#                 sentence_out_index = int(key.split('_')[1])
#                 break
#
#     ref_doc = doc_texts[ref_doc_id]
#     return update_text_doc(ref_doc, sentence_in, sentence_out_index)


def generate_updated_document(doc_texts, max_pair, ref_doc_id, rep_doc_id):
    out_index, in_index = [int(item) for item in max_pair.split('_')[1:]]
    ref_doc = sent_tokenize(doc_texts[ref_doc_id])
    rep_doc = sent_tokenize(doc_texts[rep_doc_id])
    ref_doc[out_index] = rep_doc[in_index]
    return '\n'.join(ref_doc)


def get_game_state(trec_file, current_epoch):
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


def generate_document_tfidf_files(swig_path, index_path, workingset_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    command = 'java -Djava.library.path=' + swig_path + ' -cp seo_indri_utils.jar PrepareTFIDFVectorsWS ' \
              + index_path + ' ' + workingset_file + ' ' + output_dir
    run_and_print(command)


def record_doc_similarity(doc_texts, current_epoch, similarity_file, word_embedding_model):
    ensure_dir(similarity_file)
    recent_documents = []
    for document in doc_texts:
        epoch = int(document.split('-')[1])
        if epoch == current_epoch:
            recent_documents.append(doc_texts[document])

    ensure_dir(similarity_file)
    similarity = centroid_similarity(*recent_documents, word_embedding_model)
    with open(similarity_file, 'a') as f:
        f.write(f'{current_epoch}. {similarity}\n')


def report_replacement(replacements_file, epoch, max_pair):
    ensure_dir(replacements_file)
    with open(replacements_file, 'a') as f:
        f.write(f'{epoch}. {max_pair}')
