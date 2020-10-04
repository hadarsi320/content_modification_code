from collections import defaultdict

import numpy as np


def in_group(competitor, group, dummy_bot):
    bots = ['BOT', 'DUMMY' + dummy_bot]
    return (group == 'bots' and competitor in bots) or \
           (group == 'students' and competitor not in bots) or \
           (group == 'true_bots' and competitor == 'BOT') or \
           (group == 'dummy_bots' and competitor == 'DUMMY' + dummy_bot)


def get_scaled_promotion(last_rank, current_rank):
    if last_rank == current_rank:
        return 0
    return (last_rank - current_rank) / (last_rank if last_rank > current_rank else (4 - last_rank))


def compute_average_rank(ranked_lists, competitors_lists, group):
    """
    :param trec_dir: the directory of trec files created in a competition
    :param group: who to compute for, the bots? the planted documents? the students?
    :return: an array with 4 cells where the i-th cell is the average rank in the i-th round
    """
    ranks = []
    for competition_id in ranked_lists:
        dummy_bot = competition_id.split('_')[3]
        competition = ranked_lists[competition_id]
        competitors_list = competitors_lists[competition_id]

        for competitor in competitors_list:
            if not in_group(competitor, group, dummy_bot):
                continue

            epoch_ranks = []
            for epoch in sorted(competition):
                rank = competition[epoch].index(competitor)
                epoch_ranks.append(rank)
            ranks.append(epoch_ranks)

    average_rank = np.average(ranks, axis=0)
    return average_rank


def compute_average_promotion(ranked_lists, competitors_lists, group, scaled=False):
    """
    :param ranked_lists: di
    :param group: who to compute for, the bots? the planted documents? the students?
    :param scaled: if True: return average scaled promotion, if False: return average promotion
    :return: an array with 3 cells where the i-th cell is the promotion from i-th round to the i+1-th round
    """
    rank_promotion = []
    for competition_id in ranked_lists:
        dummy_bot = competition_id.split('_')[3]
        competition = ranked_lists[competition_id]
        competitors_list = competitors_lists[competition_id]
        for competitor in competitors_list:
            if not in_group(competitor, group, dummy_bot):
                continue
            lst = []
            for last_epoch, epoch in zip(sorted(competition), sorted(competition)[1:]):
                last_rank = competition[last_epoch].index(competitor)
                rank = competition[epoch].index(competitor)
                lst.append(get_scaled_promotion(last_rank, rank) if scaled else (last_rank - rank))
            rank_promotion.append(lst)
    average_rank_promotion = np.average(rank_promotion, axis=0)
    return average_rank_promotion


    # rank_promotion = defaultdict(list)
    # for competition_id in ranked_lists:
    #     dummy_bot = competition_id.split('_')[3]
    #     competition = ranked_lists[competition_id]
    #     competitors_list = competitors_lists[competition_id]
    #
    #     for competitor in competitors_list:
    #         if not in_group(competitor, group, dummy_bot):
    #             continue
    #
    #         for last_epoch, epoch in zip(sorted(competition), sorted(competition)[1:]):
    #             last_rank = competition[last_epoch].index(competitor)
    #             rank = competition[epoch].index(competitor)
    #             rank_promotion[epoch].append(get_scaled_promotion(last_rank, rank) if scaled else last_rank - rank)
                # if last_rank > 0 or (last_rank == 0 and rank > 0):
                #     rank_promotion[last_epoch].append((last_rank - rank) / last_rank
                #                                       if scaled else last_rank - rank)

    # average_rank_promotion = [np.average(rank_promotion[epoch]) for epoch in sorted(rank_promotion)]
    # return average_rank_promotion
