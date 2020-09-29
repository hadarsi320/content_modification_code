import logging
import os
import sys
from optparse import OptionParser

import gensim

from bot_competition import create_pair_ranker, create_initial_trectext_file, create_initial_trec_file, \
    get_competitors
from create_bot_features import create_bot_features
from utils import get_model_name, get_qrid, read_trec_file, load_trectext_file

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = OptionParser()
    # Mandatory variables
    parser.add_option('--qid')
    parser.add_option('--dummy_bot_index', choices=['1', '2'])

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
    parser.add_option('--positions_file', default='./data/2of5_competition/documents.positions')

    (options, args) = parser.parse_args()
    # setting variables
    qid = options.qid
    positions_file = options.positions_file
    output_dir = options.output_dir
    label_aggregation_method = options.label_aggregation_method
    label_aggregation_b = options.label_aggregation_b
    svm_rank_c = options.svm_rank_c
    output_dir = options.output_dir
    total_rounds = options.total_rounds
    base_index = options.base_index
    merged_index = options.merged_index
    indri_path = options.indri_path

    # setting file and directory names
    trec_dir = output_dir + 'trec_files/'
    trectext_dir = output_dir + 'trectext_files/'
    raw_ds_dir = output_dir + 'raw_datasets/'
    doc_tfidf_dir = output_dir + 'document_tfidf/'
    reranking_dir = output_dir + 'reranking/'
    sentence_workingset_file = output_dir + 'document_ws.txt'
    comp_index = output_dir + 'index'
    replacements_file = output_dir + 'replacements/replacements_{}'.format(qid)

    model_path = options.svm_models_dir + get_model_name(label_aggregation_method, label_aggregation_b, svm_rank_c)
    create_pair_ranker(logger, model_path, options.label_aggregation_method,
                       options.label_aggregation_b, options.svm_rank_c, options.aggregated_data_dir,
                       options.seo_qrels_file, options.coherency_qrels_file, options.unranked_features_file,
                       options.svm_rank_scripts_dir)

    comp_trectext_file = create_initial_trectext_file(logger, options.trectext_file, trectext_dir, qid)
    comp_trec_file = create_initial_trec_file(logger, positions_file=positions_file, qid=qid,
                                              output_dir=trec_dir)
    word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(options.embedding_model_file, binary=True,
                                                                           limit=700000)

    for epoch in range(1, 4):  # in the paper they only ran the bot for 3 rounds
        print('\n{} Starting round {}\n'.format('#' * 8, epoch))
        bots, students = get_competitors(comp_trec_file, options.dummy_bot_index, qid, epoch)
        for bot in bots:
            qrid = get_qrid(qid, epoch)
            raw_ds_file = raw_ds_dir + 'raw_ds_out_' + qrid + '.txt'
            features_file = output_dir + 'final_features/features_{}.dat'.format(qrid)

            ranked_list = read_trec_file(comp_trec_file)
            doc_texts = load_trectext_file(comp_trectext_file)
            ref_index = bots[bot]
            top_docs_index = min(3, ref_index)
            premature_end = create_bot_features(logger, qrid, ref_index, top_docs_index, ranked_list, doc_texts, output_dir,
                                                word_embedding_model, mode=options.mode, raw_ds_file=raw_ds_file,
                                                doc_tfidf_dir=doc_tfidf_dir, index_path=comp_index)
            if premature_end:
                complete_sim_file(similarity_file, total_rounds)
                break

            ranking_file = generate_predictions(logger, model_path, options.svm_rank_scripts_dir, output_dir, features_file)
            max_pair = get_highest_ranked_pair(features_file, ranking_file)
            record_replacement(replacements_file, epoch, max_pair)

            updated_document = generate_updated_document(doc_texts, max_pair,
                                                         ref_doc_id=get_doc_id(epoch, qid, loser_id),
                                                         rep_doc_id=get_doc_id(epoch, qid, winner_id))
            winner_doc = doc_texts[get_doc_id(epoch, qid, winner_id)]
            trectext_dict = {get_doc_id(epoch + 1, qid, winner_id): winner_doc,
                             get_doc_id(epoch + 1, qid, loser_id): updated_document}
            append_to_trectext_file(comp_trectext_file, doc_texts, trectext_dict)

            # consider creating a trec file which only contains the files from the current round
            # otherwise there might be some issues in later rounds
            # TODO use multiprocessing
            ranked_list = read_raw_trec_file(comp_trec_file)
            reranked_trec_file = run_reranking(logger, updated_document, qrid, get_doc_id(epoch, qid, loser_id),
                                               doc_texts, ranked_list, indri_path, merged_index,
                                               options.swig_path, options.scripts_dir, options.stopwords_file,
                                               options.queries_text_file, options.ranklib_jar, options.rank_model,
                                               output_dir=reranking_dir)
            append_to_trec_file(comp_trec_file, reranked_trec_file)
            shutil.rmtree(reranking_dir)

            doc_texts = load_trectext_file(comp_trectext_file)
            create_index(logger, comp_trectext_file, comp_index, indri_path)
            generate_document_tfidf_files(logger, qid, epoch + 1, competitor_list, sentence_workingset_file,
                                          output_dir=doc_tfidf_dir, swig_path=options.swig_path,
                                          base_index=merged_index, new_index=comp_index)
            record_doc_similarity(logger, doc_texts, epoch + 1, similarity_file, word_embedding_model, doc_tfidf_dir)
            #update trec and trectext files
