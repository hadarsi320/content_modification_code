import os
from os import listdir

import numpy as np
from matplotlib import pyplot as plt

from bot_competition import get_competitors
from data_analysis import compute_average_rank, compute_average_promotion
from utils import read_competition_trec_file, normalize_dict_len, ensure_dir, read_positions_file


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
    functions = [lambda x: compute_average_rank(ranked_lists, competitors_lists, x, paper_data=True),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, scaled=False, paper_data=True),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, scaled=True, paper_data=True)]
    titles = ['Average Rank', 'Raw Rank Promotion', 'Scaled Rank Promotion']
    x_ticks_list = [range(1, 5), range(2, 5), range(2, 5)]

    for i in range(3):
        x_ticks = x_ticks_list[i]
        function = functions[i]
        title = titles[i]

        # results = {group: function(group) for group in groups}
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


def compare_competitions(postitions_file_1of5, trec_dir_2of5, trec_dir_3of5, show=True, plots_dir=None):
    print(postitions_file_1of5, trec_dir_2of5, trec_dir_3of5, sep='\n')

    ranked_lists_1of5 = read_positions_file(postitions_file_1of5)
    competitors_lists_1of5 = {qid: next(iter(ranked_lists_1of5[qid].values())) for qid in ranked_lists_1of5}

    trec_files = sorted(listdir(trec_dir_2of5))
    ranked_lists_2of5 = {trec_file: read_competition_trec_file(trec_dir_2of5 + trec_file) for trec_file in trec_files}
    competitors_lists_2of5 = {trec_file: get_competitors(trec_dir_2of5 + trec_file) for trec_file in trec_files}

    trec_files = sorted(listdir(trec_dir_3of5))
    ranked_lists_3of5 = {trec_file: read_competition_trec_file(trec_dir_3of5 + trec_file) for trec_file in trec_files}
    competitors_lists_3of5 = {trec_file: get_competitors(trec_dir_3of5 + trec_file) for trec_file in trec_files}

    list_of_ranked_lists = [ranked_lists_1of5, ranked_lists_2of5, ranked_lists_3of5]
    for ranked_lists in list_of_ranked_lists:
        normalize_dict_len(ranked_lists)
    list_of_competitors_lists = [competitors_lists_1of5, competitors_lists_2of5, competitors_lists_3of5]

    groups = ['students', 'bots']
    colors = ['blue', 'red', 'green']
    labels_list = [{'bots': 'Bots 1/5', 'students': 'Students 1/5'},
                   {'bots': 'Bots 2/5', 'students': 'Students 2/5'},
                   {'bots': 'Bots 3/5', 'students': 'Students 3/5'}]
    shapes_list = {'bots': 'D', 'students': 'o'}
    functions = [lambda x, y, z, w: compute_average_rank(x, y, z, paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=False, paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=True, paper_data=w)]
    titles = ['Average Rank', 'Raw Rank Promotion', 'Scaled Rank Promotion']
    x_ticks_list = [range(1, 5), range(2, 5), range(2, 5)]
    y_ticks_list = [np.arange(2.4, 4.01, 0.2), np.arange(-0.3, 0.41, 0.1), np.arange(-0.10, 0.31, 0.05)]

    fig, axs = plt.subplots(ncols=3, figsize=(22, 8))
    plt.rcParams.update({'font.size': 12})

    for i in range(3):
        ranked_lists = list_of_ranked_lists[i]
        competitors_lists = list_of_competitors_lists[i]
        labels = labels_list[i]
        color = colors[i]

        for ii in range(3):
            axis = axs[ii]
            x_ticks = x_ticks_list[ii]
            y_ticks = y_ticks_list[ii]
            function = functions[ii]
            title = titles[ii]

            results = {group: function(ranked_lists, competitors_lists, group, i == 0)
                       for group in groups}

            for group in groups:
                result = results[group]
                label = labels[group]
                shape = shapes_list[group]
                axis.plot(x_ticks, result, label=label, color=color, marker=shape)

            axis.set_title(title)
            axis.set_ylabel(title)
            axis.legend(loc='upper right')
            axis.set_xticks(x_ticks)
            axis.set_yticks(y_ticks)
            axis.set_xlabel('Round')

    if plots_dir:
        plt.savefig(plots_dir + '/competitions_comparison')
    if show:
        plt.show()


def main():
    sim_dir = 'output/2of2/run_10_3/similarity_results/'
    trec_dir_2 = 'output/2of5/run_10_12/trec_files/'
    trec_dir_3 = 'output/3of5/run_10_12/trec_files/'
    positions_file = 'data/paper_data/documents.positions'

    plots_dir = './plots'
    ensure_dir(plots_dir)

    # competition_5_analysis(trec_dir_3, show=True, plots_dir=None)
    # word_similarity_analysis(sim_dir, show=True, plots_dir=plots_dir)
    # recreate_paper_plots(positions_file, show=True)

    compare_competitions(positions_file, trec_dir_2, trec_dir_3, show=True)

    # trec_dir_2_old = 'output/2of5/run_10_3/trec_files/'
    # trec_dir_2_new = 'output/2of5/run_10_12/trec_files/'
    # # compare_competitions(positions_file, trec_dir_2_old, trec_dir_3, show=True)
    # compare_competitions(positions_file, trec_dir_2_new, trec_dir_3, show=True)


if __name__ == '__main__':
    main()
