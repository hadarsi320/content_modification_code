import os
from os import listdir

import numpy as np
from matplotlib import pyplot as plt

from bot_competition import get_competitors
from data_analysis import compute_average_rank, compute_average_promotion
from utils import read_competition_trec_file, normalize_dict_len, ensure_dirs, read_positions_file

COLORS = {'green': '#32a852', 'red': '#de1620', 'blue': '#1669de'}


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


def compare_competitions(show=True, plots_dir='plots/', **kwargs):
    list_of_ranked_lists = {}
    list_of_competitors_lists = {}
    if 'positions_files' in kwargs:
        positions_files = kwargs['positions_files']
        for key in positions_files:
            list_of_ranked_lists[key] = read_positions_file(positions_files[key])
            list_of_competitors_lists[key] = {qid: next(iter(list_of_ranked_lists[key][qid].values()))
                                              for qid in list_of_ranked_lists[key]}

    if 'trec_dirs' in kwargs:
        trec_dirs = kwargs['trec_dirs']
        for key in trec_dirs:
            trec_dir = trec_dirs[key]
            trec_files = sorted(listdir(trec_dirs[key]))
            list_of_ranked_lists[key] = {trec_file: read_competition_trec_file(trec_dir + '/' + trec_file)
                                         for trec_file in trec_files}
            list_of_competitors_lists[key] = {trec_file: get_competitors(trec_dir + '/' + trec_file)
                                              for trec_file in trec_files}

    rounds = []
    for key in list_of_ranked_lists:
        rounds.append(normalize_dict_len(list_of_ranked_lists[key]))
    assert all(num_rounds == max(rounds) for num_rounds in rounds)
    num_rounds = max(rounds)

    groups = ['students', 'bots']
    colors = ['blue', 'red', 'green']
    shapes_list = {'bots': 'D', 'students': 'o'}
    y_labels = ['Average Rank', 'Raw Rank Promotion', 'Scaled Rank Promotion']
    x_ticks_list = [range(1, num_rounds + 1), range(2, num_rounds + 1), range(2, num_rounds + 1)]
    functions = [lambda x, y, z, w: compute_average_rank(x, y, z, is_paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=False, is_paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=True, is_paper_data=w)]

    if 'axs' in kwargs:
        axs = kwargs['axs']
    else:
        _, axs = plt.subplots(ncols=3, figsize=(22, 8))
    plt.rcParams.update({'font.size': 12})

    # for i in range(3):
    for i, key in enumerate(sorted(list_of_ranked_lists)):
        ranked_lists = list_of_ranked_lists[key]
        competitors_lists = list_of_competitors_lists[key]
        color = COLORS[colors[i]]

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
                label = f'{group} {key}'
                shape = shapes_list[group]
                axis.plot(x_ticks, result, label=label, color=color, marker=shape, markersize=10)

            if 'title' in kwargs:
                axis.set_title(kwargs['title'])
            axis.set_ylabel(y_label)
            axis.legend(loc='upper right')
            axis.set_xticks(x_ticks)
            axis.set_xlabel('Round')

    if 'save_file' in kwargs and 'axs' not in kwargs:
        plt.savefig(plots_dir + kwargs['save_file'])
    if show and 'axs' not in kwargs:
        plt.show()


def main():
    paper_positions_files = {'1of5': 'data/paper_data/documents.positions'}
    paper_trec_dirs = {'2of5': 'output/2of5_10_12/trec_files/',
                       '3of5': 'output/3of5_10_12/trec_files/'}
    raifer_trec_dirs = {'1of5': 'results/1of5_10_16_16/trec_files',
                        '2of5': 'results/2of5_10_16_16/trec_files',
                        '3of5': 'results/3of5_10_16_16/trec_files'}
    plots_dir = './plots'
    ensure_dirs(plots_dir)

    compare_competitions(positions_files=paper_positions_files, trec_dirs=paper_trec_dirs, show=False,
                         save_file='Competitions Comparison Paper')
    compare_competitions(trec_dirs=raifer_trec_dirs, show=False,
                         save_file='Competitions Comparison Raifer')

    # creates a plot that compares Paper competitions to Raifer competitions
    _, axs = plt.subplots(ncols=2, nrows=3, figsize=(15, 20), sharey='row')
    paper_axs, raifer_axs = axs.transpose()
    compare_competitions(positions_files=paper_positions_files, trec_dirs=paper_trec_dirs, axs=paper_axs,
                         title='Paper')
    compare_competitions(trec_dirs=raifer_trec_dirs, axs=raifer_axs, title='Raifer')
    plt.savefig(plots_dir + '/Competition Comparison Paper vs Raifer')


if __name__ == '__main__':
    main()
