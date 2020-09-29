import logging
import os
import shutil
import sys
from optparse import OptionParser
from os.path import exists

import gensim

from bot_competition import create_initial_trec_file, \
    create_initial_trectext_file, generate_predictions, get_highest_ranked_pair, \
    get_ranked_competitors_list, generate_updated_document, append_to_trec_file, generate_document_tfidf_files, \
    record_doc_similarity, record_replacement, create_pair_ranker
from create_bot_features import run_reranking, create_bot_features
from utils import get_qrid, load_trectext_file, generate_doc_id, \
    append_to_trectext_file, read_raw_trec_file, read_trec_file, \
    complete_sim_file, create_index, get_model_name

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = OptionParser()
    # Mandatory variables
    parser.add_option('--qid')
    parser.add_option('--competitors')
    # parser.add_option('--competition_file')
    # TODO implement the use of competition file, this way

    # Optional variables that may be changed
    parser.add_option('--total_rounds', '-r', type='int', default=8)  # setting the rounds to be any more then 8 causes
    # a bug, since there are only 8 rounds in the file data/working_comp_queries.txt, so more need to be added
    parser.add_option('--output_dir', default='./output/tmp/')
    parser.add_option('--label_aggregation_method', '--agg', choices=['harmonic', 'demotion', 'weighted'],
                      default='harmonic')
    # this variable is not used if the agg method is 'demotion'
    parser.add_option('--mode', choices=['single', 'multiple'], default='single')

    # variables which are unlikely to be changed
    parser.add_option('--label_aggregation_b', '-b', default='1')
    parser.add_option('--svm_rank_c', '-c', default='0.01')
    parser.add_option('--svm_models_dir', default='./rank_svm_models/')
    parser.add_option('--aggregated_data_dir', default='./data/learning_dataset/')
    parser.add_option('--svm_rank_scripts_dir', default='./scripts/')
    parser.add_option('--seo_qrels_file', default='./data/qrels_seo_bot.txt')
    parser.add_option('--coherency_qrels_file', default='./data/coherency_aggregated_labels.txt')
    parser.add_option('--unranked_features_file', default='./data/features_bot_sorted.txt')
    parser.add_option('--trec_file', default='./trecs/trec_file_original_sorted.txt')
    parser.add_option('--trectext_file', default='./data/documents.trectext')
    parser.add_option('--rank_model', default='./rank_models/model_lambdatamart')
    parser.add_option('--ranklib_jar', default='./scripts/RankLib.jar')
    parser.add_option('--queries_text_file', default='./data/working_comp_queries.txt')
    parser.add_option('--scripts_dir', default='./scripts/')
    parser.add_option('--stopwords_file', default='./data/stopwords_list')
    parser.add_option('--indri_path', default='~/indri/')
    parser.add_option('--merged_index', default='~/work_files/merged_index/')
    parser.add_option('--base_index', default='~/work_files/clueweb_index/')
    parser.add_option("--swig_path", default='/lv_local/home/hadarsi/indri-5.6/swig/obj/java/')
    parser.add_option("--embedding_model_file", default='~/work_files/word2vec_model/word2vec_model')

    (options, args) = parser.parse_args()

    # setting variables
    qid = options.qid.zfill(3)
    competitor_list = sorted(options.competitors.split(','))
    label_aggregation_method = options.label_aggregation_method
    label_aggregation_b = options.label_aggregation_b
    svm_rank_c = options.svm_rank_c
    output_dir = options.output_dir
    total_rounds = options.total_rounds
    base_index = options.base_index
    merged_index = options.merged_index
    indri_path = options.indri_path
    doc_tfidf_dir = output_dir + 'document_tfidf/'
    reranking_dir = output_dir + 'reranking/'
    sentence_workingset_file = output_dir + 'document_ws.txt'
    comp_index = output_dir + 'index'  # change to the one above if using a single index causes any trouble
    replacements_file = output_dir + 'replacements/replacements_{}_{}'.format(qid, ','.join(competitor_list))
    similarity_file = output_dir + 'similarity_results/similarity_{}_{}.txt'.format(qid, ','.join(competitor_list))

    # creating the sentence-pair ranker
    model_path = options.svm_models_dir + get_model_name(label_aggregation_method, label_aggregation_b, svm_rank_c)
    create_pair_ranker(logger, model_path, options.label_aggregation_method,
                       options.label_aggregation_b, options.svm_rank_c, options.aggregated_data_dir,
                       options.seo_qrels_file, options.coherency_qrels_file, options.unranked_features_file,
                       options.svm_rank_scripts_dir)

    # initializing files
    for file in [replacements_file, similarity_file]:
        if exists(file):
            os.remove(file)
    comp_trec_file = create_initial_trec_file(logger, options.trec_file, output_dir + 'trec_files/', qid,
                                              competitor_list)
    comp_trectext_file = create_initial_trectext_file(logger, options.trectext_file, output_dir + 'trectext_files/',
                                                      qid, competitor_list)
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(options.embedding_model_file, binary=True,
                                                                           limit=700000)

    doc_texts = load_trectext_file(comp_trectext_file)
    create_index(logger, comp_trectext_file, comp_index, indri_path)
    generate_document_tfidf_files(logger, qid, 1, competitor_list, sentence_workingset_file, output_dir=doc_tfidf_dir,
                                  swig_path=options.swig_path, new_index=comp_index, base_index=merged_index)
    record_doc_similarity(logger, doc_texts, 1, similarity_file, word_embedding_model, doc_tfidf_dir)

    # consider setting the epoch to be 0 -> total_rounds
    for epoch in range(1, total_rounds + 1):
        print('\n{} Starting round {}\n'.format('#' * 8, epoch))

        qrid = get_qrid(qid, epoch)
        raw_ds_file = output_dir + 'raw_datasets/raw_ds_out_' + qrid + '_' + ','.join(competitor_list) + '.txt'
        features_file = output_dir + 'final_features/features_{}.dat'.format(qrid)
        winner_id, loser_id = get_ranked_competitors_list(comp_trec_file, epoch)

        ranked_list = read_trec_file(comp_trec_file)
        premature_end = create_bot_features(logger, qrid, 1, 1, ranked_list, doc_texts, output_dir,
                                            word_embedding_model, mode=options.mode, raw_ds_file=raw_ds_file,
                                            doc_tfidf_dir=doc_tfidf_dir, index_path=comp_index)
        if premature_end:
            complete_sim_file(similarity_file, total_rounds)
            break

        ranking_file = generate_predictions(logger, model_path, options.svm_rank_scripts_dir, output_dir, features_file)
        max_pair = get_highest_ranked_pair(features_file, ranking_file)
        record_replacement(replacements_file, epoch, max_pair)

        updated_document = generate_updated_document(doc_texts, max_pair,
                                                     ref_doc_id=generate_doc_id(epoch, qid, loser_id),
                                                     rep_doc_id=generate_doc_id(epoch, qid, winner_id))
        winner_doc = doc_texts[generate_doc_id(epoch, qid, winner_id)]
        trectext_dict = {generate_doc_id(epoch + 1, qid, winner_id): winner_doc,
                         generate_doc_id(epoch + 1, qid, loser_id): updated_document}
        append_to_trectext_file(comp_trectext_file, doc_texts, trectext_dict)

        # consider creating a trec file which only contains the files from the current round
        # otherwise there might be some issues in later rounds
        # TODO use multiprocessing
        ranked_list = read_raw_trec_file(comp_trec_file)
        reranked_trec_file = run_reranking(logger, updated_document, qrid, generate_doc_id(epoch, qid, loser_id),
                                           doc_texts, ranked_list, indri_path, merged_index,
                                           options.swig_path, options.scripts_dir, options.stopwords_file,
                                           options.queries_text_file, options.ranklib_jar, options.rank_model,
                                           output_dir=reranking_dir)
        append_to_trec_file(comp_trec_file, reranked_trec_file)
        shutil.rmtree(reranking_dir)

        doc_texts = load_trectext_file(comp_trectext_file)
        create_index(logger, comp_trectext_file, comp_index, indri_path)
        generate_document_tfidf_files(logger, qid, epoch + 1, competitor_list, sentence_workingset_file,
                                      output_dir=doc_tfidf_dir, swig_path=options.swig_path, new_index=comp_index,
                                      base_index=merged_index)
        record_doc_similarity(logger, doc_texts, epoch + 1, similarity_file, word_embedding_model, doc_tfidf_dir)
    # if premature_end:
    #         complete_sim_file(similarity_file, total_rounds)
    # else:
    #     doc_texts = load_trectext_file(comp_trectext_file)
    #     record_doc_similarity(doc_texts, total_rounds+1, similarity_file, word_embedding_model)
