import logging
import os
import sys
from optparse import OptionParser
from os.path import exists

from utils import get_learning_data_path, get_model_name, get_qrid, load_file
from bot_competition import generate_learning_dataset, create_model, create_initial_trec_file, \
    create_initial_trectext_file, create_features, generate_predictions, get_highest_ranked_pair, update_trec_files

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    parser = OptionParser()
    # Mandatory variables
    parser.add_option('--label_aggregation_method', '--agg', choices=['harmonic', 'demotion', 'weighted'])
    parser.add_option('--label_aggregation_b', '-b')  # this variable is optional if the agg method is 'harmonic'
    parser.add_option('--svm_rank_c', '-c')
    parser.add_option('--qid')
    parser.add_option('--competitors')
    parser.add_option('--rounds', '-r', type='int')

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
    competitor_list = sorted(options.competitors.split(','))
    output_dir = options.output_dir
    epoch = 1
    # TODO implement
    # if not in_dataset(qid, competitor_list):
    #     raise ValueError()

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

    update_trec_files(comp_trec_file, comp_trectext_file, raw_ds_file, doc_texts, epoch, qid, max_pair)

