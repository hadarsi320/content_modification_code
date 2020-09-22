from utils import get_learning_data_path, get_model_name, get_qrid, create_trectext_file, parse_trec_id, \
    generate_trec_id, append_to_trectext_file, load_file
import os
from gen_utils import run_and_print
from optparse import OptionParser
import xml.etree.ElementTree as ET
from lxml import etree
from deprecated import deprecated
from os.path import exists, basename, splitext
from create_bot_features import update_text_doc


def create_initial_trec_file(original_trec_file: str, output_dir: str, qid: str, competitors: list):
    if not exists(output_dir):
        os.makedirs(output_dir)
    qrid = get_qrid(qid, 1)
    new_trec_file = output_dir + 'trec_file_' + qid + '_[' + ','.join(competitors) + ']'
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
    new_trectext_file = output_dir + f'documents_{qid}_[{",".join(competitors)}].trectext'

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
    rankings_file = rankings_dir + '_predictions'.join(splitext(basename(features_file)))
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
                ref_doc = doc_texts[ref_doc_trec_id]
                sentence_in = line.split('\t')[3].strip('\n')
                sentence_out_index = int(key.split('_')[1])
                break

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


if __name__ == '__main__':
    parser = OptionParser()
    # Mandatory variables
    parser.add_option('--label_aggregation_method', '--agg', choices=['harmonic', 'demotion', 'weighted'])
    parser.add_option('--label_aggregation_b', '-b')  # this variable is optional if the agg method is 'harmonic'
    parser.add_option('--svm_rank_c', '-c')
    parser.add_option('--qid')
    parser.add_option('--competitors')

    # Optional variables
    parser.add_option('--svm_models_dir', default='./rank_svm_models/')
    parser.add_option('--aggregated_data_dir', default='./data/learning_dataset/')
    parser.add_option('--svm_rank_scripts_dir', default='./scripts/')
    parser.add_option('--seo_qrels_file', default='./data/qrels_seo_bot.txt')
    parser.add_option('--coherency_qrels_file', default='./data/coherency_aggregated_labels.txt')
    parser.add_option('--unranked_features_file', default='./data/features_bot_sorted.txt')
    parser.add_option('--trec_file', default='./trecs/trec_file_original_sorted.txt')
    parser.add_option('--trectext_file', default='./data/documents.trectext')
    parser.add_option('--output_dir', default='./tmp/')

    (options, args) = parser.parse_args()
    label_aggregation_method = options.label_aggregation_method
    label_aggregation_b = options.label_aggregation_b
    svm_rank_c = options.svm_rank_c
    qid = options.qid.zfill(3)
    competitor_list = options.competitors.split(',')
    output_dir = options.output_dir
    epoch = 1

    model_name = get_model_name(label_aggregation_method, label_aggregation_b, svm_rank_c)
    model_path = options.svm_models_dir + model_name
    if not exists(options.svm_models_dir + model_name):
        learning_data_dir = options.aggregated_data_dir + 'feature_sets/'
        learning_data_path = get_learning_data_path(learning_data_dir, label_aggregation_method,
                                                    label_aggregation_b)
        if not exists(learning_data_path):
            generate_learning_dataset(options.aggregated_data_dir, label_aggregation_method, options.seo_qrels_file,
                                      options.coherency_qrels_file, options.unranked_features_file)
        create_model(options.svm_rank_scripts_dir, options.svm_models_dir, model_name, learning_data_path, svm_rank_c)

    comp_trec_file = create_initial_trec_file(options.trec_file, output_dir + 'trec_files/', qid, competitor_list)
    comp_trectext_file = create_initial_trectext_file(options.trectext_file, output_dir + 'trectext_files/', qid,
                                                      competitor_list)
    doc_texts = load_file(comp_trectext_file)
    qrid = get_qrid(qid, epoch)
    raw_ds_file = output_dir + 'raw_ds_out_' + qid + '_' + ','.join(competitor_list) + '.txt'
    features_file = output_dir + 'final_features/features_{}.dat'.format(qrid)

    create_features(qrid, comp_trec_file, comp_trectext_file, raw_ds_file)
    ranking_file = generate_predictions(model_path, options.svm_rank_scripts_dir, output_dir, features_file)
    max_pair = get_highest_ranked_pair(features_file, ranking_file)

    winner_id, loser_id = get_game_state(comp_trec_file)
    new_loser_doc = generate_updated_document(max_pair, raw_ds_file, doc_texts)
    winner_doc = doc_texts[generate_trec_id(epoch, qid, winner_id)]
    trectext_dict = {generate_trec_id(epoch+1, qid, winner_id): winner_doc,
                     generate_trec_id(epoch+1, qid, loser_id): new_loser_doc}
    append_to_trectext_file(comp_trectext_file, trectext_dict)

    # update_trec_file()
    # update_trectext_file()

