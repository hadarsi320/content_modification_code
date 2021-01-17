from collections import defaultdict

import numpy as np

from utils import general_utils as utils
from utils.readers import TrecReader


def in_group(competitor, group, bots):
    dummy_bots = [bot for bot in bots if bot.startswith('DUMMY')]
    return (group == 'bots' and competitor in bots) or \
           (group == 'students' and competitor not in bots) or \
           (group == 'true_bots' and competitor == 'BOT') or \
           (group == 'dummy_bots' and competitor in dummy_bots) or \
           (group == 'planted' and competitor.startswith('DUMMY')) or \
           (group == 'actual_students' and not competitor.startswith('DUMMY') and competitor != 'BOT')


def get_scaled_promotion(last_rank, current_rank, max_rank):
    if last_rank == current_rank:
        return 0
    return (last_rank - current_rank) / ((last_rank - 1) if last_rank > current_rank else (max_rank - last_rank))


def compute_average_rank(trec_reader: TrecReader, group):
    """
    :param trec_reader:
    :param competitors_lists:
    :param group: who to compute for, the bots? the planted documents? the students?
    :return: an array with 4 cells where the i-th cell is the average rank in the i-th round
    """
    ranks = []
    for qid in trec_reader.queries():
        bots = ['BOT']
        pid_list = [pid for pid in trec_reader.get_pids(qid) if in_group(pid, group, bots)]

        for pid in pid_list:
            epoch_ranks = []
            for epoch in trec_reader.epochs():
                epoch_ranks.append(trec_reader.get_player_rank(epoch, qid, pid) + 1)
            ranks.append(epoch_ranks)

    average_rank = np.average(ranks, axis=0)
    return average_rank


def compute_average_promotion(trec_reader: TrecReader, group, scaled=False):
    """
    :param trec_reader: di
    :param group: who to compute for, the bots? the planted documents? the students?
    :param scaled: if True: return average scaled promotion, if False: return average promotion
    :return: an array with 3 cells where the i-th cell is the promotion from i-th round to the i+1-th round
    """
    max_rank = trec_reader.max_rank()
    epochs = trec_reader.epochs()
    rank_promotion = defaultdict(list)
    for qid in trec_reader.queries():
        bots = ['BOT']
        pid_list = [pid for pid in trec_reader.get_pids(qid) if in_group(pid, group, bots)]

        for pid in pid_list:
            for last_epoch, epoch in zip(epochs, epochs[1:]):
                last_rank = trec_reader.get_player_rank(last_epoch, qid, pid) + 1
                rank = trec_reader.get_player_rank(epoch, qid, pid) + 1
                rank_promotion[epoch].append(
                    get_scaled_promotion(last_rank, rank, max_rank) if scaled else last_rank - rank)

    average_rank_promotion = [np.average(rank_promotion[epoch]) for epoch in epochs[1:]]
    return average_rank_promotion


def cumpute_atd(ranked_lists):
    """
    Computes the average top duration of students and bots
    """
    bots_td, students_td = [], []
    for competition_id in ranked_lists:
        bots = competition_id.split('_')[3].split(',')
        competition = ranked_lists[competition_id]

        duration = 1
        last_top_player = None
        for epoch in sorted(competition):
            top_player = competition[epoch][0]
            if last_top_player is not None:
                if top_player == last_top_player:
                    duration += 1
                else:
                    if last_top_player in bots:
                        bots_td.append(duration)
                    else:
                        students_td.append(duration)
                    duration = 1
            last_top_player = top_player

    students_atd = np.average(students_td) if len(students_td) > 0 else 0
    bots_atd = np.average(bots_td) if len(bots_td) > 0 else 0
    return students_atd, bots_atd


def term_difference(text_1, text_2, terms, opposite=False):
    return utils.count_occurrences(text_1, terms, opposite) - utils.count_occurrences(text_2, terms, opposite)
