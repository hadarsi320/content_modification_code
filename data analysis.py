from collections import defaultdict
from os import listdir

import numpy as np
import matplotlib.pyplot as plt

from bot_competition import get_competitors
from plotting import plot
from utils import read_competition_trec_file, normalize_dict_len


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

    average_rank_promotion = np.average(ranks, axis=0)
    return average_rank_promotion


def compute_average_promotion(ranked_lists, competitors_lists, group, scaled=False):
    """
    :param ranked_lists: di
    :param group: who to compute for, the bots? the planted documents? the students?
    :param scaled: if True: return average scaled promotion, if False: return average promotion
    :return: an array with 3 cells where the i-th cell is the promotion from i-th round to the i+1-th round
    """
    rank_promotion = defaultdict(list)
    for competition_id in ranked_lists:
        dummy_bot = competition_id.split('_')[3]
        competition = ranked_lists[competition_id]
        competitors_list = competitors_lists[competition_id]

        for competitor in competitors_list:
            if not in_group(competitor, group, dummy_bot):
                continue

            last_epoch = None
            for epoch in sorted(competition):
                if last_epoch is not None:
                    last_rank = competition[last_epoch].index(competitor)
                    rank = competition[epoch].index(competitor)
                    rank_promotion[epoch].append(get_scaled_promotion(last_rank, rank) if scaled else last_rank - rank)
                    # if last_rank > 0 or (last_rank == 0 and rank > 0):
                    #     rank_promotion[last_epoch].append((last_rank - rank) / last_rank
                    #                                       if scaled else last_rank - rank)
                last_epoch = epoch

    average_rank_promotion = [np.average(rank_promotion[epoch]) for epoch in sorted(rank_promotion)]
    return average_rank_promotion


def main():
    trec_dir = '/lv_local/home/hadarsi/pycharm_projects/content_modification_code/output/run_2of5_10_1/trec_files/'
    trec_files = sorted(listdir(trec_dir))
    ranked_lists = {trec_file: read_competition_trec_file(trec_dir + trec_file) for trec_file in trec_files}
    normalize_dict_len(ranked_lists)
    competitors_lists = {trec_file: get_competitors(trec_dir + trec_file) for trec_file in trec_files}

    # Average Rank
    x_ticks = [1, 2, 3, 4]
    average_rank_bots = compute_average_rank(ranked_lists, competitors_lists, 'bots')
    average_rank_true_bots = compute_average_rank(ranked_lists, competitors_lists, 'true_bots')
    average_rank_dummy_bots = compute_average_rank(ranked_lists, competitors_lists, 'dummy_bots')
    average_rank_students = compute_average_rank(ranked_lists, competitors_lists, 'students')

    fig, axs = plt.subplots(ncols=2, sharey=True)
    fig.suptitle('Average Rank')
    axs[0].plot(x_ticks, average_rank_bots, label='All Bots')
    axs[0].plot(x_ticks, average_rank_students, label='Student Documents')
    axs[1].plot(x_ticks, average_rank_true_bots, label='True Bots')
    axs[1].plot(x_ticks, average_rank_dummy_bots, label='Dummy Bots')

    axs[0].set_ylabel('Average Rank')
    for ax in axs:
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Round')
    plt.show()


    # Raw Rank Promotion
    x_ticks = [2, 3, 4]
    raw_promotion_bots = compute_average_promotion(ranked_lists, competitors_lists, 'bots', scaled=False)
    raw_promotion_true_bots = compute_average_promotion(ranked_lists, competitors_lists, 'true_bots', scaled=False)
    raw_promotion_dummy_bots = compute_average_promotion(ranked_lists, competitors_lists, 'dummy_bots', scaled=False)
    raw_promotion_students = compute_average_promotion(ranked_lists, competitors_lists, 'students', scaled=False)

    fig, axs = plt.subplots(ncols=2, sharey=True)
    fig.suptitle('Average Raw Rank Promotion')
    axs[0].plot(x_ticks, raw_promotion_bots, label='All Bots')
    axs[0].plot(x_ticks, raw_promotion_students, label='Student Documents')
    axs[1].plot(x_ticks, raw_promotion_true_bots, label='True Bots')
    axs[1].plot(x_ticks, raw_promotion_dummy_bots, label='Dummy Bots')

    axs[0].set_ylabel('Average Raw Rank Promotion')
    for ax in axs:
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Round')
    plt.show()

    # Scaled Rank Promotion
    x_ticks = [2, 3, 4]
    scaled_promotion_bots = compute_average_promotion(ranked_lists, competitors_lists, 'bots', scaled=True)
    scaled_promotion_true_bots = compute_average_promotion(ranked_lists, competitors_lists, 'true_bots', scaled=True)
    scaled_promotion_dummy_bots = compute_average_promotion(ranked_lists, competitors_lists, 'dummy_bots', scaled=True)
    scaled_promotion_students = compute_average_promotion(ranked_lists, competitors_lists, 'students', scaled=True)

    fig, axs = plt.subplots(ncols=2, sharey=True)
    fig.suptitle('Average Scaled Rank Promotion')
    axs[0].plot(x_ticks, scaled_promotion_bots, label='All Bots')
    axs[0].plot(x_ticks, scaled_promotion_students, label='Student Documents')
    axs[1].plot(x_ticks, scaled_promotion_true_bots, label='True Bots')
    axs[1].plot(x_ticks, scaled_promotion_dummy_bots, label='Dummy Bots')

    axs[0].set_ylabel('Average Scaled Rank Promotion')
    for ax in axs:
        ax.legend()
        ax.set_xticks(x_ticks)
        ax.set_xlabel('Round')
    plt.show()


if __name__ == '__main__':
    main()
