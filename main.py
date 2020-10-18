import logging
import os
import pickle
import shutil
import sys
from optparse import OptionParser
from os.path import exists

from bot_competition import create_pair_ranker, create_initial_trectext_file, create_initial_trec_file, \
    get_rankings, find_fastest_climbing_document
from bot_competition import generate_predictions, get_highest_ranked_pair, \
    generate_updated_document, update_trec_file, generate_document_tfidf_files, \
    record_doc_similarity, record_replacement
from create_bot_features import create_bot_features
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
    comp_trec_file = create_initial_trec_file(output_dir=trec_dir, qid=qid, trec_file=full_trec_file,
                                              bots=competitor_list, only_bots=True)
    comp_trectext_file = create_initial_trectext_file(output_dir=trectext_dir, qid=qid, trectext_file=trectext_file,
                                                      bots=competitor_list, only_bots=True)

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
        cant_append = create_bot_features(qrid=qrid, ref_index=1, top_docs_index=1, ranked_lists=ranked_lists,
                                          doc_texts=doc_texts, output_dir=output_dir,
                                          word_embed_model=word_embedding_model, mode=run_mode, raw_ds_file=raw_ds_file,
                                          doc_tfidf_dir=doc_tfidf_dir, base_index=base_index, new_index=comp_index,
                                          documents_workingset_file=document_workingset_file, swig_path=swig_path,
                                          queries_file=queries_xml_file, final_features_file=features_file)
        if cant_append:
            complete_sim_file(similarity_file, total_rounds)
            break

        # ranking the pairs
        ranking_file = generate_predictions(pair_rank_model, svm_rank_scripts_dir, predictions_dir,
                                            features_file)

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


def run_general_competition(mode, qid, competitors, bots, rounds, trectext_file, output_dir,
                            document_workingset_file, indri_path, swig_path, doc_tfidf_dir, reranking_dir, trec_dir,
                            trectext_dir, raw_ds_dir, predictions_dir, final_features_dir, base_index, comp_index,
                            replacements_file, svm_rank_scripts_dir, run_mode, scripts_dir, stopwords_file,
                            queries_text_file, queries_xml_file, ranklib_jar, document_rank_model,
                            pair_rank_model, word_embedding_model, **kwargs):
    assert mode in ['paper', 'raifer']

    logger = logging.getLogger(sys.argv[0])
    original_texts = load_trectext_file(trectext_file, qid)

    comp_trectext_file = create_initial_trectext_file(trectext_file, trectext_dir, qid, bots=bots, only_bots=False)
    comp_trec_file = create_initial_trec_file(output_dir=trec_dir, qid=qid, bots=bots, only_bots=False, **kwargs)

    create_index(comp_trectext_file, new_index_name=comp_index, indri_path=indri_path)
    create_documents_workingset(document_workingset_file, 1, qid, competitors)
    generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                  swig_path=swig_path, base_index=base_index, new_index=comp_index)

    for epoch in range(1, rounds + 1):
        print('\n{} Starting round {}\n'.format('#' * 8, epoch))
        qrid = get_qrid(qid, epoch)
        ranked_lists = read_trec_file(comp_trec_file)
        doc_texts = load_trectext_file(comp_trectext_file)
        bots, students = get_rankings(comp_trec_file, bots, qid, epoch)

        new_docs = {}
        for student_id in students:
            next_doc_id = get_doc_id(epoch + 1, qid, student_id)
            new_docs[next_doc_id] = original_texts[next_doc_id]

        for bot_id in bots:
            logger.info(f'{bot_id} rank: {bots[bot_id]}')

            features_file = final_features_dir + f'features_{qrid}_{bot_id}.dat'
            raw_ds_file = raw_ds_dir + f'raw_ds_out_{qrid}_{bot_id}.txt'
            bot_doc_id = get_doc_id(epoch, qid, bot_id)
            next_doc_id = get_doc_id(epoch + 1, qid, bot_id)
            ref_index = bots[bot_id]

            if ref_index == 0:
                target = find_fastest_climbing_document(ranked_lists, qid)
                if target is not None:
                    target_doc_id = get_doc_id(epoch, qid, target)
                    cant_replace = create_bot_features(qrid=qrid, ref_index=ref_index, target_docs=[target_doc_id],
                                                       ranked_lists=ranked_lists, doc_texts=doc_texts,
                                                       output_dir=output_dir, word_embed_model=word_embedding_model,
                                                       mode=run_mode, raw_ds_file=raw_ds_file,
                                                       doc_tfidf_dir=doc_tfidf_dir,
                                                       documents_workingset_file=document_workingset_file,
                                                       base_index=base_index, new_index=comp_index, swig_path=swig_path,
                                                       queries_file=queries_xml_file, final_features_file=features_file)

                else:
                    cant_replace = True

            else:
                # Creating features
                top_docs_index = min(3, ref_index)
                cant_replace = create_bot_features(qrid=qrid, ref_index=ref_index, top_docs_index=top_docs_index,
                                                   ranked_lists=ranked_lists, doc_texts=doc_texts,
                                                   output_dir=output_dir,
                                                   word_embed_model=word_embedding_model, mode=run_mode,
                                                   raw_ds_file=raw_ds_file, doc_tfidf_dir=doc_tfidf_dir,
                                                   documents_workingset_file=document_workingset_file,
                                                   base_index=base_index, new_index=comp_index, swig_path=swig_path,
                                                   queries_file=queries_xml_file, final_features_file=features_file)

            if cant_replace:
                new_docs[next_doc_id] = doc_texts[bot_doc_id]
                logger.info('Bot {} cant replace any sentence'.format(bot_id))
                continue

            # Rank pairs
            ranking_file = generate_predictions(pair_rank_model, svm_rank_scripts_dir, predictions_dir,
                                                features_file)

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


def main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("Running %s" % ' '.join(sys.argv))

    parser = OptionParser()

    # Variables
    parser.add_option('--mode', choices=['2of2', 'paper', 'raifer'])
    parser.add_option('--qid')
    parser.add_option('--bots')

    # TODO implement the use of competition file, in order to run multiple competitions simultaneously
    # parser.add_option('--competition_file')
    parser.add_option('--total_rounds', '-r', type='int', default=10)
    parser.add_option('--output_dir', default='./output/tmp/')
    parser.add_option('--label_aggregation_method', '--agg',
                      choices=['harmonic', 'demotion', 'weighted'], default='harmonic')
    parser.add_option('--run_mode', choices=['single', 'multiple'], default='single')
    parser.add_option('--word2vec_dump')

    # Defaults
    parser.add_option('--label_aggregation_b', '-b', default='1')
    parser.add_option('--svm_rank_c', '-c', default='0.01')
    parser.add_option('--svm_models_dir', default='./rank_svm_models/')
    parser.add_option('--aggregated_data_dir', default='./data/learning_dataset/')
    parser.add_option('--svm_rank_scripts_dir', default='./scripts/')
    parser.add_option('--seo_qrels_file', default='./data/qrels_seo_bot.txt')
    parser.add_option('--coherency_qrels_file', default='./data/coherency_aggregated_labels.txt')
    parser.add_option('--unranked_features_file', default='./data/features_bot_sorted.txt')
    parser.add_option('--trec_file', default='./data/trec_file_original_sorted.txt')
    parser.add_option('--trectext_file_raifer', default='./data/documents.trectext')
    parser.add_option('--trectext_file_paper', default='./data/paper_data/documents.trectext')
    parser.add_option('--positions_file', default='./data/paper_data/documents.positions')
    parser.add_option('--rank_model', default='./rank_models/model_lambdatamart')
    parser.add_option('--ranklib_jar', default='./scripts/RankLib.jar')
    parser.add_option('--queries_text_file', default='./data/working_comp_queries_expanded.txt')
    parser.add_option('--queries_xml_file', default='./data/queries_seo_exp.xml')
    parser.add_option('--scripts_dir', default='./scripts/')
    parser.add_option('--stopwords_file', default='./data/stopwords_list')
    parser.add_option('--indri_path', default='/lv_local/home/hadarsi/indri/')
    parser.add_option('--clueweb_index', default='/lv_local/home/hadarsi/work_files/clueweb_index/')
    parser.add_option('--merged_index', default='/lv_local/home/hadarsi/work_files/merged_index/')
    parser.add_option("--swig_path", default='/lv_local/home/hadarsi/indri-5.6/swig/obj/java/')
    parser.add_option("--embedding_model_file",
                      default='/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model')

    (options, args) = parser.parse_args()

    try:
        int(options.qid)
    except ValueError:
        raise ValueError('qid {} is not an integer'.format(options.qid))
    qid = options.qid.zfill(3)

    output_dir = options.output_dir
    trec_dir = output_dir + 'trec_files/'
    trectext_dir = output_dir + 'trectext_files/'
    raw_ds_dir = output_dir + 'raw_datasets/'
    doc_tfidf_dir = output_dir + 'document_tfidf/'
    reranking_dir = output_dir + 'reranking/'
    predictions_dir = output_dir + 'predictions/'
    temp_index = output_dir + 'index'
    document_workingset_file = output_dir + 'document_ws.txt'
    final_features_dir = output_dir + 'final_features/'

    bots = sorted(options.bots.split(','))
    svm_rank_model = options.svm_models_dir + get_model_name(options.label_aggregation_method,
                                                             options.label_aggregation_b, options.svm_rank_c)
    create_pair_ranker(svm_rank_model, options.label_aggregation_method,
                       options.label_aggregation_b, options.svm_rank_c, options.aggregated_data_dir,
                       options.seo_qrels_file, options.coherency_qrels_file, options.unranked_features_file,
                       options.svm_rank_scripts_dir)

    if options.word2vec_dump is None:
        word_embedding_model = load_word_embedding_model(options.embedding_model_file)
        logger.info('Loaded word Embedding Model from file')
    else:
        word_embedding_model = pickle.load(open(options.word2vec_dump, 'rb'))
        logger.info('Loaded word Embedding Model from pickle')

    if options.mode == '2of2':
        trectext_file = options.trectext_file_raifer
        assert len(bots) == 2
        replacements_file = output_dir + 'replacements/replacements_{}_{}'.format(qid, ','.join(bots))
        similarity_file = output_dir + 'similarity_results/similarity_{}_{}.txt'.format(qid, ','.join(bots))
        for file in [replacements_file, similarity_file]:
            if exists(file):
                os.remove(file)

        run_2_bot_competition(qid, bots, trectext_file, options.trec_file, options.output_dir, options.clueweb_index,
                              temp_index, document_workingset_file, doc_tfidf_dir, reranking_dir, trec_dir,
                              trectext_dir, raw_ds_dir, predictions_dir, final_features_dir, options.swig_path,
                              options.indri_path, replacements_file, similarity_file, options.svm_rank_scripts_dir,
                              options.total_rounds, options.run_mode, options.scripts_dir, options.stopwords_file,
                              options.queries_text_file, options.queries_xml_file, options.ranklib_jar,
                              options.rank_model, svm_rank_model, word_embedding_model)
    else:
        replacements_file = output_dir + 'replacements/replacements_{}_{}'.format(qid, ','.join(bots))
        if exists(replacements_file):
            os.remove(replacements_file)
        competitors = get_competitors(qid=qid,
            trec_file=options.trec_file if options.mode == 'raifer' else options.positions_file)

        if not all([bot in competitors for bot in bots]):
            raise ValueError(f'Not all given bots are competitors in the query \n'
                             f'bots: {bots} \ncompetitors: {competitors}')

        if options.mode == 'raifer':
            trectext_file = options.trectext_file_raifer
            run_general_competition('paper', qid, competitors, bots, 7, trectext_file, output_dir,
                                    document_workingset_file, options.indri_path, options.swig_path,
                                    doc_tfidf_dir, reranking_dir, trec_dir, trectext_dir, raw_ds_dir, predictions_dir,
                                    final_features_dir, options.clueweb_index, temp_index, replacements_file,
                                    options.svm_rank_scripts_dir, options.run_mode, options.scripts_dir,
                                    options.stopwords_file, options.queries_text_file, options.queries_xml_file,
                                    options.ranklib_jar, options.rank_model, svm_rank_model, word_embedding_model,
                                    trec_file=options.trec_file)
        elif options.mode == 'paper':
            trectext_file = options.trectext_file_paper
            run_general_competition('paper', qid, competitors, bots, 3, trectext_file, output_dir,
                                    document_workingset_file, options.indri_path, options.swig_path,
                                    doc_tfidf_dir, reranking_dir, trec_dir, trectext_dir, raw_ds_dir, predictions_dir,
                                    final_features_dir, options.clueweb_index, temp_index, replacements_file,
                                    options.svm_rank_scripts_dir, options.run_mode, options.scripts_dir,
                                    options.stopwords_file, options.queries_text_file, options.queries_xml_file,
                                    options.ranklib_jar, options.rank_model, svm_rank_model, word_embedding_model,
                                    positions_file=options.positions_file)


if __name__ == '__main__':
    main()
