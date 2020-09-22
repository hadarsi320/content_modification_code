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
    epoch = 1
    qid = '017'
    qrid = get_qrid(qid, epoch)
    output_dir = './tmp/'
    features_file = output_dir + 'final_features/features_{}.dat'.format(qrid)
    ranking_file = output_dir + 'predictions/features_{}_predictions.dat'.format(qrid)

    # winner_id, loser_id = get_game_state(comp_trec_file)
    # winner_doc = get_doc_text(comp_trectext_file, generate_trec_id(epoch, qid, winner_id))
    # loser_doc = get_doc_text(comp_trectext_file, generate_trec_id(epoch, qid, loser_id))
    # d = {generate_trec_id(epoch, qid, player): get_doc_text(comp_trectext_file, generate_trec_id(epoch, qid, player))
    #      for player in [winner_id, loser_id]}
    # pass

    max_pair = get_highest_ranked_pair(features_file, ranking_file)
    print(max_pair)


