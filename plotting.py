import os
from os import listdir

import numpy as np
from matplotlib import pyplot as plt

from data_analysis import compute_average_rank, compute_average_promotion, compute_average_bot_top_duration
from utils import read_competition_trec_file, normalize_dict_len, ensure_dirs, read_positions_file, read_trec_dir, \
    get_competitors

COLORS = {'green': '#32a852', 'red': '#de1620', 'blue': '#1669de', 'orange': '#f28e02', 'purple': '#8202f2'}


def plot(data, start=0, stop=None, shape='o-', title=None, x_label=None, y_label=None, save_file=None, show=False,
         fig_size=(6, 4)):
    # plt.clf()
    plt.figure(figsize=fig_size)
    if not stop:
        stop = len(data) + start
    plt.plot(range(start, stop), data, shape)
    plt.xticks(range(start, stop))
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if save_file:
        plt.savefig(save_file)
    if show:
        plt.show()


def word_similarity_analysis(sim_dir, show=False, plots_dir=None):
    similarity_files = sorted(os.listdir(sim_dir))

    sim_lists = [], []
    for file in similarity_files:
        sim_list = [], []
        with open(sim_dir + '/' + file) as f:
            for num, line in enumerate(f):
                if num == 0:
                    continue
                for i in [0, 1]:
                    sim_list[i].append(float(line.split('\t')[i + 1]))
        for i in [0, 1]:
            sim_lists[i].append(sim_list[i])

    rounds = max(len(lst) for lst in sim_lists[0])
    matrices = [np.array([lst for lst in sim_lists[i] if len(lst) == rounds]) for i in range(2)]

    titles = ['Lexical Similarity', 'Embedding Similarity']
    plt.rcParams.update({'font.size': 14})
    fig, axs = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle('Similarity Analysis')
    for i in range(2):
        averaged_mat = np.average(matrices[i], axis=0)
        axs[i].plot(range(rounds), averaged_mat, 'o-')
        axs[i].set_xticks(range(rounds))
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Rounds')
        axs[i].set_ylabel('Cosine Similarity')

    if plot:
        plt.savefig(plots_dir + '/Similarity Plot')
    if show:
        plt.show()


def competition_5_analysis(trec_dir, show=True, plots_dir=None):
    trec_files = sorted(listdir(trec_dir))
    ranked_lists = {trec_file: read_competition_trec_file(trec_dir + trec_file) for trec_file in trec_files}
    normalize_dict_len(ranked_lists)
    competitors_lists = {trec_file: get_competitors(trec_dir + trec_file) for trec_file in trec_files}

    groups = ['bots', 'students', 'true_bots', 'dummy_bots']
    colors = {'bots': 'b', 'students': 'g', 'dummy_bots': 'r', 'true_bots': 'm'}
    labels = {'bots': 'All Bots', 'students': 'Student Documents', 'true_bots': 'True Bots', 'dummy_bots': 'Dummy Bots'}
    functions = [lambda x: compute_average_rank(ranked_lists, competitors_lists, x),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, False),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, True)]
    titles = ['Average Rank', 'Average Raw Rank Promotion', 'Average Scaled Rank Promotion']
    x_ticks_list = [range(1, 5), range(2, 5), range(2, 5)]

    for i in range(3):
        x_ticks = x_ticks_list[i]
        function = functions[i]
        title = titles[i]

        results = {group: function(group) for group in groups}
        fig, axs = plt.subplots(ncols=2, sharey='row', figsize=(13, 5))
        plt.rcParams.update({'font.size': 14})
        for j, group in enumerate(groups):
            result = results[group]
            label = labels[group]
            color = colors[group]
            fig_num = int(j / 2)
            axs[fig_num].plot(x_ticks, result, label=label, color=color)

        fig.suptitle(title)
        axs[0].set_ylabel(title)
        axs[0].set_title('Bots vs. Students')
        axs[1].set_title('Original Bots vs. Dummy Bots')
        for ax in axs:
            ax.legend()
            ax.set_xticks(x_ticks)
            ax.set_xlabel('Round')
        if plots_dir:
            plt.savefig(plots_dir + '/' + title)
        if show:
            plt.show()


def recreate_paper_plots(positions_file, show=True, plots_dir=None):
    """
    This function recreates the plots from the paper, based on the positions file from the paper
    :param positions_file:
    :param show:
    :param plots_dir:
    :return:
    """
    ranked_lists = read_positions_file(positions_file)
    normalize_dict_len(ranked_lists)
    competitors_lists = {qid: next(iter(ranked_lists[qid].values())) for qid in ranked_lists}

    groups = ['actual_students', 'planted', 'bots']
    colors = {'bots': 'b', 'actual_students': 'r', 'planted': 'g'}
    labels = {'bots': 'All Bots', 'actual_students': 'Student Documents', 'planted': 'Planted Documents'}
    functions = [lambda x: compute_average_rank(ranked_lists, competitors_lists, x, is_paper_data=True),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, scaled=False,
                                                     is_paper_data=True),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, scaled=True,
                                                     is_paper_data=True)]
    titles = ['Average Rank', 'Raw Rank Promotion', 'Scaled Rank Promotion']
    x_ticks_list = [range(1, 5), range(2, 5), range(2, 5)]

    for i in range(3):
        x_ticks = x_ticks_list[i]
        function = functions[i]
        title = titles[i]

        results = {}
        for group in groups:
            results[group] = function(group)

        print(title)
        for group in results:
            print(f'{group}: {results[group]}')
        print()

        plt.rcParams.update({'font.size': 14})
        for j, group in enumerate(groups):
            result = results[group]
            label = labels[group]
            color = colors[group]
            plt.plot(x_ticks, result, label=label, color=color)

        plt.title(title)
        plt.ylabel(title)
        plt.legend()
        plt.xticks(x_ticks)
        plt.xlabel('Round')
        if plots_dir:
            plt.savefig(plots_dir + '/' + title)
        if show:
            plt.show()


def get_optimal_yticks(results: dict, jump_len):
    max_val = max(max(results[key]) for key in results)
    min_val = min(min(results[key]) for key in results)

    max_tick = (max_val // jump_len + 4) * jump_len
    min_tick = min_val // jump_len * jump_len

    return np.arange(min_tick, max_tick, jump_len)


def compare_competitions(show=True, plots_dir='plots/', **kwargs):
    ranked_lists_dict = {}
    competitors_lists_dict = {}

    if 'positions_files' in kwargs:
        positions_files = kwargs.pop('positions_files')
        for key in positions_files:
            ranked_lists_dict[key] = read_positions_file(positions_files[key])
            competitors_lists_dict[key] = {qid: next(iter(ranked_lists_dict[key][qid].values()))
                                           for qid in ranked_lists_dict[key]}

    if 'trec_dirs' in kwargs:
        trec_dirs = kwargs.pop('trec_dirs')
        for key in trec_dirs:
            ranked_lists_dict[key], competitors_lists_dict[key] = read_trec_dir(trec_dirs[key])

    rounds = []
    for key in ranked_lists_dict:
        rounds.append(normalize_dict_len(ranked_lists_dict[key]))
    assert all(num_rounds == max(rounds) for num_rounds in rounds)
    num_rounds = max(rounds)

    groups = ['students', 'bots']
    colors = [COLORS[color] for color in ['blue', 'red', 'green', 'orange', 'purple']]
    shapes_list = {'bots': 'D', 'students': 'o'}
    y_labels = ['Average Rank', 'Raw Rank Promotion', 'Scaled Rank Promotion']
    x_ticks_list = [range(1, num_rounds + 1), range(2, num_rounds + 1), range(2, num_rounds + 1)]
    jump_lengths = [0.25, 0.1, 0.05]
    functions = [lambda x, y, z, w: compute_average_rank(x, y, z, is_paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=False, is_paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=True, is_paper_data=w)]

    if 'axs' in kwargs:
        axs = kwargs.pop('axs')
    else:
        _, axs = plt.subplots(ncols=3, figsize=(22, 8))
    plt.rcParams.update({'font.size': 12})

    for i, key in enumerate(ranked_lists_dict):
        ranked_lists = ranked_lists_dict[key]
        competitors_lists = competitors_lists_dict[key]
        color = colors[i]

        for ii in range(3):
            axis = axs[ii]
            x_ticks = x_ticks_list[ii]
            function = functions[ii]
            y_label = y_labels[ii]

            is_paper_data = 'positions_files' in kwargs and key in kwargs['positions_files']
            results = {group: function(ranked_lists, competitors_lists, group, is_paper_data)
                       for group in groups}

            for group in groups:
                result = results[group]
                label = f'{key}- {group}'
                shape = shapes_list[group]
                axis.plot(x_ticks, result, label=label, color=color, marker=shape, markersize=10)

            if 'title' in kwargs:
                axis.set_title(kwargs['title'])
            axis.set_ylabel(y_label)
            axis.legend(ncol=2, loc='upper right')
            axis.set_xticks(x_ticks)
            axis.set_yticks(get_optimal_yticks(results, jump_len=jump_lengths[ii]))
            axis.set_xlabel('Round')

    if 'save_file' in kwargs and 'axs' not in kwargs:
        plt.savefig(plots_dir + kwargs['save_file'])
    if show and 'axs' not in kwargs:
        plt.show()


def analyze_top_upgrades(trec_dirs_list, show=True, **kwargs):
    colors = [COLORS[color] for color in ['blue', 'red', 'green', 'orange', 'purple']]

    plt.figure().clear()
    for i, trec_dirs in enumerate(trec_dirs_list):
        color = colors[i]
        label = labels[i]

        list_of_ranked_lists = {}
        competitions = sorted(trec_dirs)
        for key in competitions:
            list_of_ranked_lists[key], _ = read_trec_dir(trec_dirs[key])
        average_bot_top_duration = [compute_average_bot_top_duration(list_of_ranked_lists[key])
                                    for key in list_of_ranked_lists]

        plt.plot(competitions, average_bot_top_duration, label=label, color=color)
    plt.xlabel('Competition Type')
    plt.title('Average First Rank Duration')
    plt.legend()

    if show:
        plt.show()
    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'])


def main():
    plots_dir = './plots'
    ensure_dirs(plots_dir)

    competitions_1of5 = {'vanilla': 'results/1of5_10_16_16/trec_files',
                         'acceleration': 'results/2of5_10_18_15_1st_promotion_v1/trec_files',
                         'past_top': 'results/1of5_10_19_20_past_top/trec_files',
                         'inferiors': 'results/1of5_10_19_20_highest_rated_inferiors/trec_files'}
    competitions_2of5 = {'vanilla': 'results/2of5_10_16_16/trec_files',
                         'acceleration': 'results/2of5_10_18_15_1st_promotion_v1/trec_files',
                         'past_top': 'results/2of5_10_19_20_past_top/trec_files',
                         'inferiors': 'results/2of5_10_19_20_highest_rated_inferiors/trec_files'}
    competitions_3of5 = {'vanilla': 'results/3of5_10_16_16/trec_files',
                         'acceleration': 'results/2of5_10_18_15_1st_promotion_v1/trec_files',
                         'past_top': 'results/3of5_10_19_20_past_top/trec_files',
                         'inferiors': 'results/3of5_10_20_01_highest_rated_inferiors/trec_files'}
    competitions_4of5 = {'vanilla': 'results/4of5_10_16_16/trec_files',
                         'acceleration': 'results/2of5_10_18_15_1st_promotion_v1/trec_files',
                         'past_top': 'results/4of5_10_19_20_past_top/trec_files',
                         'inferiors': 'results/4of5_10_20_01_highest_rated_inferiors/trec_files'}
    competitions_5of5 = {'vanilla': 'results/5of5_10_16_16/trec_files',
                         'acceleration': 'results/2of5_10_18_15_1st_promotion_v1/trec_files',
                         'past_top': 'results/5of5_10_19_20_past_top/trec_files',
                         'inferiors': 'results/5of5_10_20_05_highest_rated_inferiors/trec_files'}

    # competitions_list = [competitions_1of5, competitions_2of5, competitions_3of5, competitions_4of5, competitions_5of5]
    competitions_list = [competitions_1of5, competitions_2of5, competitions_3of5]

    _, axs = plt.subplots(ncols=3, nrows=len(competitions_list), figsize=(20, 20))
    for i, ax in enumerate(axs):
        competitions = competitions_list[i]
        compare_competitions(trec_dirs=competitions, axs=ax, title=f'{i+1} bots out of 5', show=False)
    # plt.savefig(plots_dir + '/Competitions Comparison of Top Refinement Methods')
    plt.show()

    # analyze_top_upgrades(competitions_list, show=True)
                         # savefig=plots_dir + '/Average First Place Duration Comparison')


if __name__ == '__main__':
    main()
