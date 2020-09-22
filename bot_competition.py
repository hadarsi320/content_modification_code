import os
import xml.etree.ElementTree as ET
from os.path import exists, basename, splitext

from lxml import etree

from create_bot_features import update_text_doc, run_reranking
from gen_utils import run_and_print
from utils import get_qrid, create_trectext_file, parse_trec_id, \
    generate_trec_id, append_to_trectext_file


def create_initial_trec_file(original_trec_file: str, output_dir: str, qid: str, competitors: list):
    if not exists(output_dir):
        os.makedirs(output_dir)
    qrid = get_qrid(qid, 1)
    new_trec_file = output_dir + 'trec_file_' + qid + '_' + ','.join(competitors) + ''
    with open(original_trec_file, 'r') as trec_file:
        with open(new_trec_file, 'w') as f:
            for line in trec_file:
                line_qrid = line.split()[0]
                if int(line_qrid) > int(qrid):
                    break
                competitor = line.split()[2].split('-')[-1]
                if qrid == line_qrid and competitor in competitors:
                    f.write(line)
    return new_trec_file


def create_initial_trectext_file(original_trectext_file, output_dir, qid, competitors):
    if not exists(output_dir):
        os.makedirs(output_dir)
    doc_list = [f'ROUND-01-{qid}-{competitor}' for competitor in competitors]
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
                if name not in doc_list:
                    break
            elif att.tag == 'TEXT':
                docs[name] = att.text

    create_trectext_file(docs, new_trectext_file)
    return new_trectext_file


def create_features(qrid, trec_file, trectext_file, raw_ds_fname, mode='single', ref_index=1, top_docs_index=1):
    command = f'python create_bot_features.py --mode={mode} --qrid={qrid} --ref_index={ref_index} ' \
              f'--top_docs_index={top_docs_index} --trec_file={trec_file} --trectext_file={trectext_file}' \
              f' --raw_ds_out={raw_ds_fname}'
    run_and_print(command)


def generate_learning_dataset(output_dir, label_aggregation_method, seo_qrels, coherency_qrels, feature_fname):
    command = 'python dataset_creator.py ' + output_dir + ' ' + label_aggregation_method + ' ' + seo_qrels + ' ' + \
              coherency_qrels + ' ' + feature_fname
    run_and_print(command)


def create_model(svm_rank_scripts_dir, models_dir, model_name, learning_data, svm_rank_c):
    if not exists(models_dir):
        os.makedirs(models_dir)
    model_path = models_dir + model_name
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
        pairs = [line.split('# ')[1].strip('\n') for line in f if len(line) > 0]

    with open(predictions_file, 'r') as f:
        scores = [float(line) for line in f if len(line) > 0]

    max_pair, _ = max(zip(pairs, scores), key=lambda x: x[1])
    return max_pair


def generate_updated_document(max_pair, raw_ds_file, doc_texts):
    # TODO use update texts for the multiple competitions version
    with open(raw_ds_file) as f:
        for line in f:
            pair = line.split('\t')[1]
            key = pair.split('$')[1]
            if key == max_pair:
                ref_doc_trec_id = pair.split('$')[0]
                sentence_in = line.split('\t')[3].strip('\n')
                sentence_out_index = int(key.split('_')[1])
                break

    ref_doc = doc_texts[ref_doc_trec_id]
    return update_text_doc(ref_doc, sentence_in, sentence_out_index)


def get_game_state(trec_file):
    competitors_ranked_list = []
    with open(trec_file, 'r') as f:
        for line in f:
            trec_id = line.split()[2]
            competitors_ranked_list.append(parse_trec_id(trec_id)[2])
    return competitors_ranked_list


def get_doc_text(doctext_file, trec_id):
    xml_parser = etree.XMLParser(recover=True)
    tree = ET.parse(doctext_file, parser=xml_parser)
    root = tree.getroot()
    last_trec_id = ''
    for doc in root:
        for att in doc:
            if att.tag == 'DOCNO':
                last_trec_id = att.text
            elif att.tag == 'TEXT' and (last_trec_id == trec_id):
                return att.text
    raise ValueError('No document was found in path {} with the trec_id {}'
                     .format(doctext_file, trec_id))


def update_trec_files(logger, trec_file, trectext_file, raw_ds_file, doc_texts, epoch, qid, max_pair):
    winner_id, loser_id = get_game_state(trec_file)
    updated_document = generate_updated_document(max_pair, raw_ds_file, doc_texts)
    winner_doc = doc_texts[generate_trec_id(epoch, qid, winner_id)]
    trectext_dict = {generate_trec_id(epoch + 1, qid, winner_id): winner_doc,
                     generate_trec_id(epoch + 1, qid, loser_id): updated_document}
    append_to_trectext_file(trectext_file, trectext_dict)

    t = run_reranking(logger, )


