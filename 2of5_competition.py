import logging
import os
import sys
from optparse import OptionParser

from bot_competition import create_pair_ranker

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = OptionParser()
    # Mandatory variables
    parser.add_option('--qid')

    parser.add_option('--output_dir', default='./output/tmp/')
    parser.add_option('--label_aggregation_method', '--agg', choices=['harmonic', 'demotion', 'weighted'],
                      default='harmonic')
    parser.add_option('--mode', choices=['single', 'multiple'], default='single')

    parser.add_option('--label_aggregation_b', '-b', default='1')
    parser.add_option('--svm_rank_c', '-c', default='0.01')
    parser.add_option('--svm_models_dir', default='./rank_svm_models/')
    parser.add_option('--aggregated_data_dir', default='./data/learning_dataset/')
    parser.add_option('--svm_rank_scripts_dir', default='./scripts/')
    parser.add_option('--seo_qrels_file', default='./data/qrels_seo_bot.txt')
    parser.add_option('--coherency_qrels_file', default='./data/coherency_aggregated_labels.txt')
    parser.add_option('--unranked_features_file', default='./data/features_bot_sorted.txt')

    (options, args) = parser.parse_args()

    create_pair_ranker(logger, options.svm_models_dir, options.label_aggregation_b, options.svm_rank_c, options.aggregated_data_dir,
                       options.seo_qrels_file, options.coherency_qrels_file, options.unranked_features_file,
                       options.svm_rank_scripts_dir)


    for epoch in range(1, 4):  # in the paper they only ran the bot for 3 rounds
        pass
