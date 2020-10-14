from collections import defaultdict

import numpy as np


def in_group(competitor, group, dummy_bot):
    dummy_bots = ['DUMMY1', 'DUMMY2'] if dummy_bot == 'both' else \
        (['DUMMY' + dummy_bot] if dummy_bot in ['1', '2'] else [])
    bots = ['BOT'] + dummy_bots
    planted_documents = ['DUMMY_1', 'DUMMY_2']
    return (group == 'bots' and competitor in bots) or \
           (group == 'students' and competitor not in bots) or \
           (group == 'true_bots' and competitor == 'BOT') or \
           (group == 'dummy_bots' and competitor in dummy_bots) or \
           (group == 'planted' and competitor in planted_documents) or \
           (group == 'actual_students' and competitor not in planted_documents and competitor != 'BOT')


def get_scaled_promotion(last_rank, current_rank):
    if last_rank == current_rank:
        return 0
    return (last_rank - current_rank) / ((last_rank - 1) if last_rank > current_rank else (5 - last_rank))
    # return (last_rank - current_rank) / max(last_rank, 5 - last_rank)


def compute_average_rank(ranked_lists, competitors_lists, group, paper_data=False):
    """
    :param ranked_lists:
    :param competitors_lists:
    :param group: who to compute for, the bots? the planted documents? the students?
    :param paper_data:
    :return: an array with 4 cells where the i-th cell is the average rank in the i-th round
    """
    ranks = []
    for competition_id in ranked_lists:
        dummy_bot = None if paper_data else competition_id.split('_')[3]
        competition = ranked_lists[competition_id]

        competitors_list = competitors_lists[competition_id]

        for competitor in competitors_list:
            if not in_group(competitor, group, dummy_bot):
                continue

            epoch_ranks = []
            for epoch in sorted(competition):
                rank = competition[epoch].index(competitor) + 1
                epoch_ranks.append(rank)
            ranks.append(epoch_ranks)

        # competition_ranks = defaultdict(list)
        # for epoch in sorted(competition):
        #     for i, pid in enumerate(competition[epoch]):
        #         if in_group(pid, group, dummy_bot):
        #             competition_ranks[pid].append(i+1)
        # for pid in competition_ranks:
        #     ranks.append(competition_ranks[pid])

    average_rank = np.average(ranks, axis=0)
    return average_rank


def compute_average_promotion(ranked_lists, competitors_lists, group, scaled=False, paper_data=False):
    """
    :param ranked_lists: di
    :param group: who to compute for, the bots? the planted documents? the students?
    :param scaled: if True: return average scaled promotion, if False: return average promotion
    :return: an array with 3 cells where the i-th cell is the promotion from i-th round to the i+1-th round
    """
    # rank_promotion = []
    # for competition_id in ranked_lists:
    #     dummy_bot = None if paper_data else competition_id.split('_')[3]
    #     competition = ranked_lists[competition_id]
    #     competitors_list = competitors_lists[competition_id]
    #     for competitor in competitors_list:
    #         if not in_group(competitor, group, dummy_bot):
    #             continue
    #         lst = []
    #         for last_epoch, epoch in zip(sorted(competition), sorted(competition)[1:]):
    #             last_rank = competition[last_epoch].index(competitor)+1
    #             rank = competition[epoch].index(competitor)+1
    #             lst.append(get_scaled_promotion(last_rank, rank) if scaled else (last_rank - rank))
    #         rank_promotion.append(lst)
    # average_rank_promotion = np.average(rank_promotion, axis=0)
    # return average_rank_promotion
    epochs = sorted(ranked_lists[next(iter(ranked_lists))])
    rank_promotion = defaultdict(list)
    for competition_id in ranked_lists:
        dummy_bot = None if paper_data else competition_id.split('_')[3]
        competition = ranked_lists[competition_id]
        competitors_list = competitors_lists[competition_id]

        for competitor in competitors_list:
            if not in_group(competitor, group, dummy_bot):
                continue

            for last_epoch, epoch in zip(epochs, epochs[1:]):
                last_rank = competition[last_epoch].index(competitor) + 1
                rank = competition[epoch].index(competitor) + 1
                # if last_rank == 1 and group == 'bots':
                #     continue
                rank_promotion[epoch].append(get_scaled_promotion(last_rank, rank) if scaled else last_rank - rank)

    average_rank_promotion = [np.average(rank_promotion[epoch]) for epoch in epochs[1:]]
    return average_rank_promotion
