import logging
import os
import sys
from optparse import OptionParser
from os.path import exists

import gensim

from create_bot_features import run_reranking
from utils import get_learning_data_path, get_model_name, get_qrid, load_trectext_file, generate_trec_id, \
    append_to_trectext_file, read_raw_trec_file, create_sentence_workingset, create_index
from bot_competition import generate_learning_dataset, create_model, create_initial_trec_file, \
    create_initial_trectext_file, create_features, generate_predictions, get_highest_ranked_pair, \
    get_game_state, generate_updated_document, append_to_trec_file, generate_document_tfidf_files, \
    document_doc_similarity

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    parser = OptionParser()
    # Mandatory variables
    parser.add_option('--qid')
    parser.add_option('--competitors')

    # Optional variables
    parser.add_option('--total_rounds', '-r', type='int', default=25)
    parser.add_option('--label_aggregation_method', '--agg', choices=['harmonic', 'demotion', 'weighted'],
                      default='harmonic')
    parser.add_option('--label_aggregation_b', '-b',
                      default='1')  # this variable is not used if the agg method is 'demotion'
    parser.add_option('--svm_rank_c', '-c', default='0.01')
    parser.add_option('--svm_models_dir', default='./rank_svm_models/')
    parser.add_option('--aggregated_data_dir', default='./data/learning_dataset/')
    parser.add_option('--svm_rank_scripts_dir', default='./scripts/')
    parser.add_option('--seo_qrels_file', default='./data/qrels_seo_bot.txt')
    parser.add_option('--coherency_qrels_file', default='./data/coherency_aggregated_labels.txt')
    parser.add_option('--unranked_features_file', default='./data/features_bot_sorted.txt')
    parser.add_option('--trec_file', default='./trecs/trec_file_original_sorted.txt')
    parser.add_option('--trectext_file', default='./data/documents.trectext')
    parser.add_option('--output_dir', default='./output/')
    parser.add_option('--rank_model', default='./rank_models/model_lambdatamart')
    parser.add_option('--ranklib_jar', default='./scripts/RankLib.jar')
    parser.add_option('--queries_text_file', default='./data/working_comp_queries.txt')
    parser.add_option('--scripts_dir', default='./scripts/')
    parser.add_option('--stopwords_file', default='./data/stopwords_list')
    parser.add_option('--indri_path', default='~/indri/')
    parser.add_option('--index_path', default='~/work_files/merged_index/')
    parser.add_option("--swig_path", default='/lv_local/home/hadarsi/indri-5.6/swig/obj/java/')
    parser.add_option("--reranking_output_dir", default='./output/reranking_output/')
    parser.add_option("--embedding_model_file", default='~/work_files/word2vec_model/word2vec_model')
    # parser.add_option('--', default='')

    (options, args) = parser.parse_args()
    label_aggregation_method = options.label_aggregation_method
    label_aggregation_b = options.label_aggregation_b
    svm_rank_c = options.svm_rank_c
    qid = options.qid.zfill(3)
    competitor_list = sorted(options.competitors.split(','))
    output_dir = options.output_dir
    doc_tfidf_dir = output_dir + 'document_tfidf_dir/'
    sentence_workingset_file = output_dir + 'document_ws.txt'
    comp_index = output_dir + 'index_dir/index_{}_{}'.format(qid, ','.join(competitor_list))
    similarity_file = './similarity_results/similarity_{}_{}.txt'.format(qid, ','.join(competitor_list))
    if os.path.exists(similarity_file):
        os.remove(similarity_file)
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(options.embedding_model_file, binary=True,
                                                                           limit=700000)
    # TODO implement some qid fool-proofing protocol
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

    for epoch in range(1, options.total_rounds + 1):
        print('\n#########Starting round {}\n'.format(epoch))  # (press enter to begin)
        doc_texts = load_trectext_file(comp_trectext_file)
        document_doc_similarity(doc_texts, epoch, similarity_file, word_embedding_model)
        qrid = get_qrid(qid, epoch)
        raw_ds_file = output_dir + 'raw_datasets/raw_ds_out_' + qrid + '_' + ','.join(competitor_list) + '.txt'
        features_file = output_dir + 'final_features/features_{}.dat'.format(qrid)
        winner_id, loser_id = get_game_state(comp_trec_file, epoch)

        create_index(comp_trectext_file, comp_index, options.indri_path)
        create_sentence_workingset(sentence_workingset_file, epoch, qid, competitor_list)
        generate_document_tfidf_files(options.swig_path, comp_index, sentence_workingset_file,
                                      doc_tfidf_dir)

        create_features(qrid, comp_trec_file, comp_trectext_file, raw_ds_file, doc_tfidf_dir, comp_index)
        ranking_file = generate_predictions(model_path, options.svm_rank_scripts_dir, output_dir, features_file)
        max_pair = get_highest_ranked_pair(features_file, ranking_file)

        print('########################## max pair {}'.format(max_pair))
        updated_document = generate_updated_document(max_pair, raw_ds_file, doc_texts)
        winner_doc = doc_texts[generate_trec_id(epoch, qid, winner_id)]
        trectext_dict = {generate_trec_id(epoch + 1, qid, winner_id): winner_doc,
                         generate_trec_id(epoch + 1, qid, loser_id): updated_document}
        append_to_trectext_file(comp_trectext_file, trectext_dict)

        ranked_list = read_raw_trec_file(comp_trec_file)
        # consider creating a file which only contains the files from the current round
        # otherwise there might be some issues in later rounds
        reranked_trec_file = run_reranking(logger, updated_document, qrid, generate_trec_id(epoch, qid, loser_id),
                                           doc_texts, ranked_list, options.indri_path, options.index_path,
                                           options.swig_path, options.scripts_dir, options.stopwords_file,
                                           options.queries_text_file, options.ranklib_jar, options.rank_model,
                                           options.reranking_output_dir, 'new_index', 'specific_ws',
                                           'new_trectext_file', 'new_feature_file', 'feature_dir/', 'trec_file',
                                           'score_file')
        append_to_trec_file(comp_trec_file, reranked_trec_file)
