import gensim
import pickle
from time import time
from tqdm import tqdm
from optparse import OptionParser
from nltk import sent_tokenize

from create_bot_features import *
from bot_competition import *
from utils import *

if __name__ == '__main__':
    # comp_trec_file = '/lv_local/home/hadarsi/pycharm_projects/content_modification_code/tmp/trec_files' \
    #                  '/trec_file_010_[40,49]'
    # comp_trectext_file = '/lv_local/home/hadarsi/pycharm_projects/content_modification_code/tmp/trectext_files' \
    #                      '/documents_010_[40,49].trectext'
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    epoch = 1
    qid = '017'
    qrid = get_qrid(qid, epoch)
    trec_file = './tmp/trec_files/trec_file_017_01,22'
    output_dir = './tmp/reranking_test/'
    raw_ds_file = './tmp/raw_ds_out_017_01,22.txt'
    features_file = './tmp/final_features/features_{}.dat'.format(qrid)
    new_feature_file = './tmp/reranking_test/features_{}.dat'.format(qrid)
    ranking_file = './tmp/predictions/features_{}_predictions.dat'.format(qrid)
    trectext_file = './tmp/trectext_files/documents_017_01,22.trectext'

    max_pair = get_highest_ranked_pair(features_file, ranking_file)
    doc_texts = load_file(trectext_file)
    ranked_lists = read_raw_trec_file(trec_file)

    with open(raw_ds_file) as f:
        for line in f:
            pair = line.split('\t')[1]
            key = pair.split('$')[1]
            if key == max_pair:
                ref_doc_trec_id = pair.split('$')[0]
                sentence_in = line.split('\t')[3].strip('\n')
                sentence_out_index = int(key.split('_')[1])
                break

    run_reranking(logger, output_dir +'new_index', sentence_in, output_dir + 'specific_ws', ref_doc_trec_id,
                  sentence_out_index, output_dir + 'new_trectext', ranked_lists, new_feature_file,
                  output_dir +'feature_dir/', output_dir +'trec_file', output_dir +'score_file', '~/', 'indri/',
                  '/lv_local/home/hadarsi/work_files/merged_index/', './scripts/',
                  '/lv_local/home/hadarsi/indri-5.6/swig/obj/java/', './data/stopwords_list',
                  './data/working_comp_queries.txt', './scripts/RankLib.jar', './rank_models/model_lambdatamart')

