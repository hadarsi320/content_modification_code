import logging
import os
import pickle
import shutil
import sys

from bot.bot_competition import create_pair_ranker, create_initial_trectext_file, create_initial_trec_file, \
    get_rankings, get_target_documents, generate_predictions, get_highest_ranked_pair, generate_updated_document, \
    update_trec_file, generate_document_tfidf_files, record_replacement, replacement_validation
from bot.create_bot_features import create_bot_features
from bot.create_bot_features import run_reranking
from utils.general_utils import get_doc_id, update_trectext_file, create_index, create_documents_workingset, \
    load_word_embedding_model, get_competitors, ensure_dirs, get_model_name, get_qrid, read_trectext_file
from utils.readers import TrecReader


# @deprecated("This function is outdated and is incompatible with many of the current ")
# def run_2_bot_competition(qid, competitor_list, trectext_file, full_trec_file, output_dir, base_index, comp_index,
#                           document_workingset_file, doc_tfidf_dir, reranking_dir, trec_dir, trectext_dir, raw_ds_dir,
#                           predictions_dir, final_features_dir, swig_path, indri_path, replacements_file,
#                           similarity_file, svm_rank_scripts_dir, total_rounds, scripts_dir, stopwords_file,
#                           queries_text_file, queries_xml_file, ranklib_jar, document_rank_model, pair_rank_model,
#                           word_embedding_model):
#     # initalizing the trec and trectext files specific to this competition
#     comp_trec_file = create_initial_trec_file(output_dir=trec_dir, qid=qid, trec_file=full_trec_file,
#                                               bots=competitor_list, only_bots=True)
#     comp_trectext_file = create_initial_trectext_file(output_dir=trectext_dir, qid=qid, trectext_file=trectext_file,
#                                                       bots=competitor_list, only_bots=True)
#
#     doc_texts = read_trectext_file(comp_trectext_file)
#     create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
#     create_documents_workingset(document_workingset_file, competitor_list, qid, 1)
#     generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
#                                   swig_path=swig_path, base_index=base_index, new_index=comp_index)
#     record_doc_similarity(doc_texts, 1, similarity_file, word_embedding_model, doc_tfidf_dir)
#
#     for epoch in range(1, total_rounds + 1):
#         print('\n{} Starting round {}\n'.format('#' * 8, epoch))
#         qrid = get_qrid(qid, epoch)
#         raw_ds_file = raw_ds_dir + 'raw_ds_out_{}_{}.txt'.format(qrid, ','.join(competitor_list))
#         features_file = final_features_dir + 'features_{}_{}.dat'.format(qrid, ','.join(competitor_list))
#
#         ranked_lists = utils.read_trec_file(comp_trec_file)
#         winner_doc_id, loser_doc_id = ranked_lists[str(epoch).zfill(2)][qid]
#
#         # creating features
#         cant_append = create_bot_features(qrid=qrid, ref_index=1, ranked_lists=ranked_lists,
#                                           doc_texts=doc_texts, output_dir=output_dir,
#                                           word_embed_model=word_embedding_model, raw_ds_file=raw_ds_file,
#                                           doc_tfidf_dir=doc_tfidf_dir, base_index=base_index, new_index=comp_index,
#                                           documents_workingset_file=document_workingset_file, swig_path=swig_path,
#                                           queries_file=queries_xml_file, final_features_file=features_file)
#         if cant_append:
#             complete_sim_file(similarity_file, total_rounds)
#             break
#
#         # ranking the pairs
#         ranking_file = generate_predictions(pair_rank_model, svm_rank_scripts_dir, predictions_dir, features_file)
#
#         # creating the new document
#         rep_doc_id, out_index, in_index = get_highest_ranked_pair(features_file, ranking_file)
#         record_replacement(replacements_file, epoch, loser_doc_id, rep_doc_id, out_index, in_index)
#         updated_document = generate_updated_document(doc_texts, ref_doc_id=loser_doc_id, rep_doc_id=winner_doc_id,
#                                                      out_index=out_index, in_index=in_index)
#         winner_doc = doc_texts[winner_doc_id]
#
#         # updating the trectext file
#         new_docs = {get_next_doc_id(winner_doc_id): winner_doc,
#                     get_next_doc_id(loser_doc_id): updated_document}
#         update_trectext_file(comp_trectext_file, doc_texts, new_docs)
#
#         # updating the index
#         create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
#
#         # TODO use multiprocessing
#         # updating the trec file
#         reranked_trec_file = run_reranking(qrid, comp_trec_file, base_index, comp_index, swig_path,
#                                            scripts_dir, stopwords_file, queries_text_file, ranklib_jar,
#                                            document_rank_model, output_dir=reranking_dir)
#         update_trec_file(comp_trec_file, reranked_trec_file)
#
#         # removing the reranking dir so that the ranking does not get reused by mistake
#         shutil.rmtree(reranking_dir)
#
#         # creating document tfidf vectors (+ recording doc similarity)
#         doc_texts = read_trectext_file(comp_trectext_file)
#         create_documents_workingset(document_workingset_file, competitor_list, qid, epoch + 1)
#         generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
#                                       swig_path=swig_path, base_index=base_index, new_index=comp_index)
#         record_doc_similarity(doc_texts, epoch + 1, similarity_file, word_embedding_model, doc_tfidf_dir)

def run_general_competition(qid, competitors, bots, rounds, top_refinement, validation_method, trectext_file,
                            output_dir,
                            document_workingset_file, indri_path, swig_path, doc_tfidf_dir, reranking_dir, trec_dir,
                            trectext_dir, raw_ds_dir, predictions_dir, final_features_dir, base_index, comp_index,
                            replacements_file, svm_rank_scripts_dir, scripts_dir, stopwords_file,
                            queries_text_file, queries_xml_file, ranklib_jar, document_rank_model, pair_ranker,
                            top_ranker, word_embedding_model, alternation_classifier, rep_val_dir, **kwargs):
    logger = logging.getLogger(sys.argv[0])
    original_texts = read_trectext_file(trectext_file, qid)

    comp_trectext_file = create_initial_trectext_file(trectext_file, trectext_dir, qid, bots=bots, only_bots=False)
    comp_trec_file = create_initial_trec_file(output_dir=trec_dir, qid=qid, bots=bots, only_bots=False, **kwargs)

    create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
    create_documents_workingset(document_workingset_file, competitors=competitors, qid=qid, epochs=[1])
    generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                  swig_path=swig_path, base_index=base_index, new_index=comp_index)

    past_targets = {}
    for epoch in range(1, rounds + 1):
        print('\n{} Starting round {}\n'.format('#' * 8, epoch))
        qrid = get_qrid(qid, epoch)
        trec_reader = TrecReader(trec_file=comp_trec_file)
        trec_texts = read_trectext_file(comp_trectext_file)
        bot_rankings, student_rankings = get_rankings(comp_trec_file, bots, qid, epoch)

        new_docs = {}
        for student_id in student_rankings:
            next_doc_id = get_doc_id(epoch + 1, qid, student_id)
            new_docs[next_doc_id] = original_texts[next_doc_id]

        for bot_id in bot_rankings:
            logger.info(f'{bot_id} rank: {bot_rankings[bot_id] + 1}')

            features_file = final_features_dir + f'features_{qrid}_{bot_id}.dat'
            raw_ds_file = raw_ds_dir + f'raw_ds_out_{qrid}_{bot_id}.txt'

            bot_doc_id = get_doc_id(epoch, qid, bot_id)
            next_doc_id = get_doc_id(epoch + 1, qid, bot_id)
            bot_rank = bot_rankings[bot_id]

            target_documents = get_target_documents(epoch, qid, bot_id, bot_rank, trec_reader, past_targets,
                                                    top_refinement)
            past_targets[qid] = target_documents
            if target_documents is not None:
                # Creating features
                cant_replace = create_bot_features(qrid=qrid, ref_index=bot_rank, target_docs=target_documents,
                                                   ranked_lists=trec_reader, doc_texts=trec_texts,
                                                   output_dir=output_dir, word_embed_model=word_embedding_model,
                                                   raw_ds_file=raw_ds_file, doc_tfidf_dir=doc_tfidf_dir,
                                                   documents_workingset_file=document_workingset_file,
                                                   base_index=base_index, new_index=comp_index, swig_path=swig_path,
                                                   queries_file=queries_xml_file, final_features_file=features_file)
            else:
                cant_replace = True

            if cant_replace:
                new_docs[next_doc_id] = trec_texts[bot_doc_id]
                logger.info('Bot {} cant replace any sentence'.format(bot_id))
                continue

            # Rank pairs
            ranker = top_ranker if bot_rank == 0 else pair_ranker
            ranking_file = generate_predictions(ranker, svm_rank_scripts_dir, predictions_dir, features_file)

            # Find highest ranked pair
            rep_doc_id, out_index, in_index, features = get_highest_ranked_pair(features_file, ranking_file)

            old_doc = trec_texts[bot_doc_id]
            new_doc = generate_updated_document(trec_texts, ref_doc_id=bot_doc_id, rep_doc_id=rep_doc_id,
                                                out_index=out_index, in_index=in_index)

            if bot_rank == 0:
                # reconsider replacement
                replacement_valid = replacement_validation(next_doc_id, old_doc, new_doc, qid,
                                                           epoch, validation_method, queries_xml_file, trec_reader,
                                                           trec_texts,
                                                           alternation_classifier, word_embedding_model, stopwords_file,
                                                           rep_val_dir, base_index, indri_path, swig_path)
            else:
                replacement_valid = True

            if replacement_valid:
                # Replace sentence
                record_replacement(replacements_file, epoch, bot_doc_id, rep_doc_id, out_index, in_index, features)
                new_docs[next_doc_id] = new_doc
            else:
                # Keep document from last round
                new_docs[next_doc_id] = trec_texts[bot_doc_id]

        # updating the trectext file
        update_trectext_file(comp_trectext_file, trec_texts, new_docs)

        # updating the index, workingset file and tfidf files
        create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
        create_documents_workingset(document_workingset_file, competitors=competitors, qid=qid, epochs=[epoch + 1])
        generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                      swig_path=swig_path, base_index=base_index, new_index=comp_index)

        # updating the  the trec file
        reranked_trec_file = run_reranking(qrid, comp_trec_file, base_index, comp_index, swig_path,
                                           scripts_dir, stopwords_file, queries_text_file, ranklib_jar,
                                           document_rank_model, output_dir=reranking_dir)
        update_trec_file(comp_trec_file, reranked_trec_file)
        shutil.rmtree(reranking_dir)
    shutil.rmtree(comp_index)


def competition_setup(mode, qid: str, bots: list, top_refinement, validation_method, output_dir='output/tmp/',
                      mute=False, **kwargs):
    embedding_model_file = '/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model'
    alternation_classifier_pickle = 'classifiers/alteration_classifier.pkl'
    clueweb_index = '/lv_local/home/hadarsi/work_files/clueweb_index/'
    swig_path = '/lv_local/home/hadarsi/indri-5.6/swig/obj/java/'
    coherency_qrels_file = 'data/coherency_aggregated_labels.txt'
    queries_text_file = 'data/working_comp_queries_expanded.txt'
    trectext_file_paper = 'data/paper_data/documents.trectext'
    unranked_features_file = 'data/features_bot_sorted.txt'
    positions_file = 'data/paper_data/documents.positions'
    trec_file = 'data/trec_file_original_sorted.txt'
    trectext_file_raifer = 'data/documents.trectext'
    aggregated_data_dir = 'data/learning_dataset/'
    rank_model = 'rank_models/model_lambdatamart'
    queries_xml_file = 'data/queries_seo_exp.xml'
    indri_path = '/lv_local/home/hadarsi/indri/'
    seo_qrels_file = 'data/qrels_seo_bot.txt'
    stopwords_file = 'data/stopwords_list'
    ranklib_jar = 'scripts/RankLib.jar'
    svm_rank_scripts_dir = 'scripts/'
    svm_models_dir = 'rank_models/'
    scripts_dir = 'scripts/'

    pair_ranker_args = ('harmonic', 1)

    ensure_dirs(output_dir)
    document_workingset_file = output_dir + 'document_ws.txt'
    rep_val_dir = output_dir + 'replacement_evaluation/'
    final_features_dir = output_dir + 'final_features/'
    doc_tfidf_dir = output_dir + 'document_tfidf/'
    trectext_dir = output_dir + 'trectext_files/'
    predictions_dir = output_dir + 'predictions/'
    reranking_dir = output_dir + 'reranking/'
    raw_ds_dir = output_dir + 'raw_datasets/'
    trec_dir = output_dir + 'trec_files/'
    # competition_index = output_dir + 'index_' + qid + '_' + ','.join(bots)
    competition_index = output_dir + 'index'

    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.CRITICAL + 1 if mute else logging.INFO)
    logger.info("Running {}".format(' '.join(sys.argv)))

    pair_ranker = svm_models_dir + get_model_name(pair_ranker_args)
    if not os.path.exists(pair_ranker):
        create_pair_ranker(pair_ranker, pair_ranker_args, aggregated_data_dir,
                           seo_qrels_file, coherency_qrels_file, unranked_features_file,
                           svm_rank_scripts_dir)

    if 'top_ranker_args' in kwargs:
        top_ranker_args = kwargs.pop('top_ranker_args')
        top_ranker = svm_models_dir + get_model_name(top_ranker_args)
        if not os.path.exists(top_ranker):
            create_pair_ranker(top_ranker, top_ranker_args, aggregated_data_dir,
                               seo_qrels_file, coherency_qrels_file, unranked_features_file,
                               svm_rank_scripts_dir)
    else:
        top_ranker = pair_ranker

    # load word2vec model
    if 'word2vec_dump' in kwargs:
        word2vec_dump = kwargs.pop('word2vec_dump')
        word_embedding_model = pickle.load(open(word2vec_dump, 'rb'))
        logger.info('Loaded word Embedding Model from pickle')
    else:
        word_embedding_model = load_word_embedding_model(embedding_model_file)
        logger.info('Loaded word Embedding Model from file')

    alternation_classifier = pickle.load(open(alternation_classifier_pickle, 'rb'))

    if mode == '2of2':
        trectext_file = trectext_file_raifer
        assert len(bots) == 2
        replacements_file = output_dir + 'replacements/replacements_{}_{}'.format(qid, ','.join(bots))
        similarity_file = output_dir + 'similarity_results/similarity_{}_{}.txt'.format(qid, ','.join(bots))
        for file in [replacements_file, similarity_file]:
            if os.path.exists(file):
                os.remove(file)

        run_2_bot_competition(qid, bots, trectext_file, trec_file, output_dir, clueweb_index,
                              competition_index, document_workingset_file, doc_tfidf_dir, reranking_dir, trec_dir,
                              trectext_dir, raw_ds_dir, predictions_dir, final_features_dir, swig_path,
                              indri_path, replacements_file, similarity_file, svm_rank_scripts_dir,
                              10, scripts_dir, stopwords_file, queries_text_file, queries_xml_file,
                              ranklib_jar, rank_model, pair_ranker, top_ranker, word_embedding_model)

    else:
        replacements_file = output_dir + 'replacements/replacements_' + '_'.join([qid, ','.join(bots)])
        if os.path.exists(replacements_file):
            os.remove(replacements_file)
        competitors = get_competitors(qid=qid, trec_file=(trec_file if mode == 'raifer' else positions_file))

        if mode == 'raifer':
            trectext_file = trectext_file_raifer
            kwargs = dict(trec_file=trec_file)
            rounds = 7
        elif mode == 'goren':
            trectext_file = trectext_file_paper
            kwargs = dict(positions_file=positions_file)
            rounds = 3
        else:
            raise ValueError('Illegal mode given')

        if not all([bot in competitors for bot in bots]):
            raise ValueError(f'Not all given bots are competitors in the query \n'
                             f'bots: {bots} \ncompetitors: {competitors}')

        run_general_competition(qid, competitors, bots, rounds, top_refinement, validation_method, trectext_file,
                                output_dir, document_workingset_file, indri_path, swig_path, doc_tfidf_dir,
                                reranking_dir, trec_dir, trectext_dir, raw_ds_dir, predictions_dir, final_features_dir,
                                clueweb_index, competition_index, replacements_file, svm_rank_scripts_dir, scripts_dir,
                                stopwords_file, queries_text_file, queries_xml_file, ranklib_jar, rank_model,
                                pair_ranker, top_ranker, word_embedding_model, alternation_classifier, rep_val_dir,
                                **kwargs)


if __name__ == '__main__':
    import constants

    # competition_setup(mode='goren', qid='051', bots=['BOT'],
    #                   top_refinement=constants.ACCELERATION,
    #                   validation_method=constants.PREDICTION)
    competition_setup(mode='goren', qid='051', bots=['BOT'],
                      top_refinement=constants.ACCELERATION,
                      validation_method=constants.OPTIMISTIC)

    # parser = OptionParser()
    # parser.add_option('--mode', choices=['2of2', 'paper', 'raifer'])
    # parser.add_option('--qid')
    # parser.add_option('--bots')
    # parser.add_option('--top_refinement')
    # parser.add_option('--output_dir')
    # parser.add_option('--word2vec_dump')
    # (options, args) = parser.parse_args()
    #
    # arguments_dict = {}
    # if options.output_dir is not None:
    #     arguments_dict['output_dir'] = options.output_dir
    # if options.word2vec_dump is not None:
    #     arguments_dict['word2vec_dump'] = options.word2vec_dump
    # if options.top_ranker_args is not None:
    #     arguments_dict['top_ranker_args'] = options.top_ranker_args.split('_')
    #
    # start = time()
    # competition_setup(options.mode, options.qid.zfill(3), options.bots.split(','), options.top_refinement,
    #                   **arguments_dict)
    #
    # print('\n\n\n')
    # print(f'Total Time {time() - start}')
    # timings = gen_utils.timings
    # results = [(key, len(timings[key]), np.average(timings[key]), np.var(timings[key])) for key in timings]
    # for key, length, ave, var in sorted(results, key=lambda x: x[2]):
    #     print('key: {} | len: {} | average value: {} | variance: {}'
    #           .format(key, length, ave, var))
