import logging
import os
import pickle
import shutil
import sys
from collections import defaultdict
from optparse import OptionParser
from os.path import exists

from bot_competition import create_pair_ranker, create_initial_trectext_file, create_initial_trec_file, \
    get_rankings, get_target_documents, assert_bot_input, get_competition_index
from bot_competition import generate_predictions, get_highest_ranked_pair, \
    generate_updated_document, update_trec_file, generate_document_tfidf_files, \
    record_doc_similarity, record_replacement
from create_bot_features import setup_feature_creation
from create_bot_features import run_reranking
from utils import get_doc_id, \
    update_trectext_file, complete_sim_file, create_index, create_documents_workingset, get_next_doc_id, \
    load_word_embedding_model, get_competitors
from utils import get_model_name, get_qrid, read_trec_file, load_trectext_file


def run_2_bot_competition(qid, competitor_list, trectext_file, full_trec_file, output_dir, base_index, comp_index,
                          document_workingset_file, doc_tfidf_dir, reranking_dir, trec_dir, trectext_dir, raw_ds_dir,
                          predictions_dir, final_features_dir, swig_path, indri_path, replacements_file,
                          similarity_file, svm_rank_scripts_dir, total_rounds, run_mode, scripts_dir, stopwords_file,
                          queries_text_file, queries_xml_file, ranklib_jar, document_rank_model, pair_rank_model,
                          word_embedding_model):
    # initalizing the trec and trectext files specific to this competition
    comp_trec_file = create_initial_trec_file(output_dir=trec_dir, qid_list=qid, trec_file=full_trec_file,
                                              bots_dict=competitor_list, only_bots=True)
    comp_trectext_file = create_initial_trectext_file(output_dir=trectext_dir, qid_list=qid,
                                                      trectext_file=trectext_file,
                                                      bots_dict=competitor_list, only_bots=True)

    doc_texts = load_trectext_file(comp_trectext_file)
    create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
    create_documents_workingset(document_workingset_file, 1, qid, competitor_list)
    generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                  swig_path=swig_path, base_index=base_index, new_index=comp_index)
    record_doc_similarity(doc_texts, 1, similarity_file, word_embedding_model, doc_tfidf_dir)

    # TODO set the epoch to be 0 -> total_rounds
    for epoch in range(1, total_rounds + 1):
        print('\n{} Starting round {}\n'.format('#' * 8, epoch))
        qrid = get_qrid(qid, epoch)
        raw_ds_file = raw_ds_dir + 'raw_ds_out_{}_{}.txt'.format(qrid, ','.join(competitor_list))
        features_file = final_features_dir + 'features_{}_{}.dat'.format(qrid, ','.join(competitor_list))

        ranked_lists = read_trec_file(comp_trec_file)
        winner_doc_id, loser_doc_id = ranked_lists[str(epoch).zfill(2)][qid]

        # creating features
        cant_append = setup_feature_creation(ref_indices=1, ranked_lists=ranked_lists,
                                             doc_texts=doc_texts, output_dir=output_dir,
                                             word_embed_model=word_embedding_model, mode=run_mode,
                                             raw_ds_file=raw_ds_file,
                                             doc_tfidf_dir=doc_tfidf_dir, base_index=base_index, new_index=comp_index,
                                             documents_workingset_file=document_workingset_file, swig_path=swig_path,
                                             queries_file=queries_xml_file, final_features_file=features_file)
        if cant_append:
            complete_sim_file(similarity_file, total_rounds)
            break

        # ranking the pairs
        ranking_file = generate_predictions(pair_rank_model, svm_rank_scripts_dir, predictions_dir, features_file)

        # creating the new document
        rep_doc_id, out_index, in_index = get_highest_ranked_pair(features_file, ranking_file)
        record_replacement(replacements_file, epoch, loser_doc_id, rep_doc_id, out_index, in_index)
        updated_document = generate_updated_document(doc_texts, ref_doc_id=loser_doc_id, rep_doc_id=winner_doc_id,
                                                     out_index=out_index, in_index=in_index)
        winner_doc = doc_texts[winner_doc_id]

        # updating the trectext file
        new_trectext_dict = {get_next_doc_id(winner_doc_id): winner_doc,
                             get_next_doc_id(loser_doc_id): updated_document}
        update_trectext_file(comp_trectext_file, doc_texts, new_trectext_dict)

        # updating the index
        create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)

        # TODO use multiprocessing
        # updating the trec file
        reranked_trec_file = run_reranking(qrid, comp_trec_file, base_index, comp_index, swig_path,
                                           scripts_dir, stopwords_file, queries_text_file, ranklib_jar,
                                           document_rank_model, output_dir=reranking_dir)
        update_trec_file(comp_trec_file, reranked_trec_file)

        # removing the reranking dir so that the ranking does not get reused by mistake
        shutil.rmtree(reranking_dir)

        # creating document tfidf vectors (+ recording doc similarity)
        doc_texts = load_trectext_file(comp_trectext_file)
        create_documents_workingset(document_workingset_file, epoch + 1, qid, competitor_list)
        generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                      swig_path=swig_path, base_index=base_index, new_index=comp_index)
        record_doc_similarity(doc_texts, epoch + 1, similarity_file, word_embedding_model, doc_tfidf_dir)


def run_general_competition(competition_index, qid_list, competitors, bots_dict, rounds, top_refinement, trectext_file,
                            output_dir,
                            document_workingset_file, indri_path, swig_path, doc_tfidf_dir, reranking_dir, trec_dir,
                            trectext_dir, raw_ds_dir, predictions_dir, final_features_dir, base_index, comp_index,
                            replacements_file, svm_rank_scripts_dir, scripts_dir, stopwords_file,
                            queries_text_file, queries_xml_file, ranklib_jar, document_rank_model, pair_rank_model,
                            word_embedding_model, **kwargs):
    logger = logging.getLogger(sys.argv[0])
    original_texts = load_trectext_file(trectext_file, qid_list)

    comp_trectext_file = create_initial_trectext_file(trectext_file, trectext_dir, competition_index, qid_list,
                                                      bots_dict)
    comp_trec_file = create_initial_trec_file(trec_dir, competition_index, qid_list, bots_dict, **kwargs)

    create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
    create_documents_workingset(document_workingset_file, 1, qid_list, competitors)
    generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                  swig_path=swig_path, base_index=base_index, new_index=comp_index)

    for epoch in range(1, rounds + 1):
        print('\n{} Starting round {}\n'.format('#' * 8, epoch))
        qrid_list = [get_qrid(qid, epoch) for qid in qid_list]
        ranked_lists = read_trec_file(comp_trec_file)
        doc_texts = load_trectext_file(comp_trectext_file)

        features_file = final_features_dir + f'features_{competition_index}_{epoch}.dat'
        raw_ds_file = raw_ds_dir + f'raw_ds_out_{competition_index}_{epoch}.txt'

        new_docs = {}
        ref_indices = defaultdict(dict)
        target_documents = defaultdict(dict)

        for qid in qid_list:
            bot_rankings, student_rankings = get_rankings(ranked_lists, epoch, qid, bots_dict[qid])
            for student_id in student_rankings:
                next_doc_id = get_doc_id(epoch + 1, qid, student_id)
                new_docs[next_doc_id] = original_texts[next_doc_id]

            for bot_id in bot_rankings:
                logger.info(f'{bot_id} rank: {bot_rankings[bot_id] + 1}')

                bot_doc_id = get_doc_id(epoch, qid, bot_id)
                next_doc_id = get_doc_id(epoch + 1, qid, bot_id)
                ref_indices[qid][bot_id] = bot_rankings[bot_id]

                if ref_indices[qid][bot_id] == 0:
                    target_documents[qid][bot_id] = get_target_documents(top_refinement, qid, epoch, ranked_lists)

                else:
                    top_docs_index = min(3, ref_indices[qid][bot_id])
                    target_documents[qid][bot_id] = ranked_lists[str(epoch).zfill(2)][qid][:top_docs_index]

                if target_documents[qid][bot_id] is None:
                    new_docs[next_doc_id] = doc_texts[bot_doc_id]
                    target_documents[qid].pop(bot_id)

        setup_feature_creation(qid_list, epoch, ref_indices=ref_indices, target_documents=target_documents,
                               ranked_lists=ranked_lists, doc_texts=doc_texts, output_dir=output_dir,
                               word_embed_model=word_embedding_model, raw_ds_file=raw_ds_file,
                               doc_tfidf_dir=doc_tfidf_dir, documents_workingset_file=document_workingset_file,
                               base_index=base_index, new_index=comp_index, swig_path=swig_path,
                               queries_file=queries_xml_file, final_features_file=features_file)

        # Rank pairs
        ranking_file = generate_predictions(pair_rank_model, svm_rank_scripts_dir, predictions_dir, features_file)

        # Find highest ranked pair
        rep_doc_id, out_index, in_index = get_highest_ranked_pair(features_file, ranking_file)

        # Replace sentence
        record_replacement(replacements_file, epoch, bot_doc_id, rep_doc_id, out_index, in_index)
        new_docs[next_doc_id] = generate_updated_document(doc_texts, ref_doc_id=bot_doc_id, rep_doc_id=rep_doc_id,
                                                          out_index=out_index, in_index=in_index)

        # updating the trectext file
        update_trectext_file(comp_trectext_file, doc_texts, new_docs)

        # updating the index, workingset file and tfidf files
        create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
        create_documents_workingset(document_workingset_file, epoch + 1, qid, competitors)
        generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                      swig_path=swig_path, base_index=base_index, new_index=comp_index)

        # updating the  the trec file
        reranked_trec_file = run_reranking(qrid, comp_trec_file, base_index, comp_index, swig_path,
                                           scripts_dir, stopwords_file, queries_text_file, ranklib_jar,
                                           document_rank_model, output_dir=reranking_dir)
        update_trec_file(comp_trec_file, reranked_trec_file)
        shutil.rmtree(reranking_dir)


def competition_setup(competition_mode, top_refinement, qid_list, bots_dict, **kwargs):
    # Default values
    output_dir = kwargs.pop('output_dir', 'output/tmp/')
    label_aggregation_method = 'harmonic'
    label_aggregation_b = 1
    svm_rank_c = 0.01
    total_rounds = 10
    svm_models_dir = 'rank_svm_models/'
    aggregated_data_dir = 'data/learning_dataset/'
    svm_rank_scripts_dir = 'scripts/'
    seo_qrels_file = 'data/qrels_seo_bot.txt'
    coherency_qrels_file = 'data/coherency_aggregated_labels.txt'
    unranked_features_file = 'data/features_bot_sorted.txt'
    trec_file = 'data/trec_file_original_sorted.txt'
    trectext_file_raifer = 'data/documents.trectext'
    trectext_file_paper = 'data/paper_data/documents.trectext'
    positions_file = 'data/paper_data/documents.positions'
    rank_model = 'rank_models/model_lambdatamart'
    ranklib_jar = 'scripts/RankLib.jar'
    queries_text_file = 'data/working_comp_queries_expanded.txt'
    queries_xml_file = 'data/queries_seo_exp.xml'
    scripts_dir = 'scripts/'
    stopwords_file = 'data/stopwords_list'
    indri_path = '/lv_local/home/hadarsi/indri/'
    clueweb_index = '/lv_local/home/hadarsi/work_files/clueweb_index/'
    swig_path = '/lv_local/home/hadarsi/indri-5.6/swig/obj/java/'
    embedding_model_file = '/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model'

    # Default directory names
    document_workingset_file = output_dir + 'document_ws.txt'
    competition_files_dir = output_dir + 'competition_files/'
    final_features_dir = output_dir + 'final_features/'
    replacements_dir = output_dir + 'replacements/'
    doc_tfidf_dir = output_dir + 'document_tfidf/'
    trectext_dir = output_dir + 'trectext_files/'
    predictions_dir = output_dir + 'predictions/'
    reranking_dir = output_dir + 'reranking/'
    raw_ds_dir = output_dir + 'raw_datasets/'
    trec_dir = output_dir + 'trec_files/'
    temp_index = output_dir + 'index'

    assert run_mode in ['serial', 'parallel']
    assert competition_mode in ['2of2', 'raifer', 'paper']
    assert top_refinement in [None, 'acceleration', 'past_top', 'highest_rated_inferiors']

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))

    svm_rank_model = svm_models_dir + get_model_name(label_aggregation_method, label_aggregation_b, svm_rank_c)
    if not exists(svm_rank_model):
        create_pair_ranker(svm_rank_model, label_aggregation_method,
                           label_aggregation_b, svm_rank_c, aggregated_data_dir,
                           seo_qrels_file, coherency_qrels_file, unranked_features_file,
                           svm_rank_scripts_dir)

    # load word2vec model
    if 'word2vec_dump' in kwargs:
        word2vec_dump = kwargs['word2vec_dump']
        word_embedding_model = pickle.load(open(word2vec_dump, 'rb'))
        logger.info('Loaded word Embedding Model from pickle')
    else:
        word_embedding_model = load_word_embedding_model(embedding_model_file)
        logger.info('Loaded word Embedding Model from file')

    competition_index = get_competition_index(qid_list, bots_dict, competition_files_dir)
    replacements_file = replacements_dir + f'replacements_{competition_index}/'

    if competition_mode == '2of2':
        trectext_file = trectext_file_raifer
        assert_bot_input(competition_mode, **kwargs)
        similarity_file = output_dir + 'similarity_results/similarity_{}_{}.txt'.format(qid_list, ','.join(bots_dict))
        for file in [replacements_file, similarity_file]:
            if exists(file):
                os.remove(file)

        run_2_bot_competition(qid_list, bots_dict, trectext_file, trec_file, output_dir, clueweb_index,
                              temp_index, document_workingset_file, doc_tfidf_dir, reranking_dir, trec_dir,
                              trectext_dir, raw_ds_dir, predictions_dir, final_features_dir, swig_path,
                              indri_path, replacements_file, similarity_file, svm_rank_scripts_dir,
                              total_rounds, run_mode, scripts_dir, stopwords_file,
                              queries_text_file, queries_xml_file, ranklib_jar,
                              rank_model, svm_rank_model, word_embedding_model)

    else:
        if exists(replacements_file):
            os.remove(replacements_file)

        competitors = get_competitors(trec_file=(trec_file if competition_mode == 'raifer' else positions_file),
                                      qid_list=qid_list)

        assert_bot_input(competition_mode, qid_list, bots_dict, **kwargs, competitors=competitors)

        if competition_mode == 'raifer':
            trectext_file = trectext_file_raifer
            run_general_competition(qid_list, competitors, bots_dict, 7, top_refinement, trectext_file,
                                    output_dir, document_workingset_file, indri_path, swig_path,
                                    doc_tfidf_dir, reranking_dir, trec_dir, trectext_dir, raw_ds_dir, predictions_dir,
                                    final_features_dir, clueweb_index, temp_index, replacements_file,
                                    svm_rank_scripts_dir, stopwords_file, queries_text_file, queries_xml_file,
                                    ranklib_jar, rank_model, svm_rank_model, word_embedding_model,
                                    trec_file=trec_file, **kwargs)
        elif competition_mode == 'paper':
            trectext_file = trectext_file_paper
            run_general_competition(qid_list, competitors, bots_dict, 3, top_refinement, trectext_file,
                                    output_dir, document_workingset_file, indri_path, swig_path,
                                    doc_tfidf_dir, reranking_dir, trec_dir, trectext_dir, raw_ds_dir, predictions_dir,
                                    final_features_dir, clueweb_index, temp_index, replacements_file,
                                    svm_rank_scripts_dir, stopwords_file, queries_text_file, queries_xml_file,
                                    ranklib_jar, rank_model, svm_rank_model, word_embedding_model,
                                    positions_file=positions_file, **kwargs)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--mode', choices=['2of2', 'paper', 'raifer'])
    parser.add_option('--qid')
    parser.add_option('--bots')
    parser.add_option('--top_refinement', choices=['acceleration', 'past_top', 'highest_rated_inferiors'])

    (options, args) = parser.parse_args()
    competition_setup(competition_mode=options.mode, run_mode='serial', top_refinement=options.top_refinement,
                      qid=options.qid, bots=options.bots)
