import os
from os import listdir

import numpy as np
from matplotlib import pyplot as plt

from bot_competition import get_competitors
from data_analysis import compute_average_rank, compute_average_promotion
from utils import read_competition_trec_file, normalize_dict_len, ensure_dir


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

    # averaged_mat = np.average(matrices[0], axis=0)
    # plot(averaged_mat, title=f'Lexical Similarity',
    #      x_label='Round', y_label='Cosine Similarity', show=show,
    #      save_file=plots_dir+'/Lexical Similarity' if plots_dir else None)
    #
    # averaged_mat = np.average(matrices[1], axis=0)
    # plot(averaged_mat, title=f'Embedding Similarity',
    #      x_label='Round', y_label='Cosine Similarity', show=show,
    #      save_file=plots_dir+'/Embedding Similarity' if plots_dir else None)

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
        plt.savefig(plots_dir+'/Similarity Plot')
    if show:
        plt.show()


def competition_2of5_analysis(trec_dir, show=True, plots_dir=None):
    groups = ['bots', 'students', 'true_bots', 'dummy_bots']
    colors = {'bots': 'b', 'students': 'g', 'dummy_bots': 'r', 'true_bots': 'm'}
    labels = {'bots': 'All Bots', 'students': 'Student Documents', 'true_bots': 'True Bots', 'dummy_bots': 'Dummy Bots'}
    functions = [lambda x: compute_average_rank(ranked_lists, competitors_lists, x),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, False),
                 lambda x: compute_average_promotion(ranked_lists, competitors_lists, x, True)]
    titles = ['Average Rank', 'Average Raw Rank Promotion', 'Average Scaled Rank Promotion']
    x_ticks_list = [range(1, 5), range(2, 5), range(2, 5)]

    trec_files = sorted(listdir(trec_dir))
    ranked_lists = {trec_file: read_competition_trec_file(trec_dir + trec_file) for trec_file in trec_files}
    normalize_dict_len(ranked_lists)
    competitors_lists = {trec_file: get_competitors(trec_dir + trec_file) for trec_file in trec_files}

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
            fig_num = int(j/2)
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


def main():
    plots_dir = './plots'
    ensure_dir(plots_dir)

    trec_dir = 'output/2of5/run_10_3/trec_files/'
    competition_2of5_analysis(trec_dir, show=True, plots_dir=plots_dir)

    sim_dir = 'output/2of2/run_10_3/similarity_results/'
    # word_similarity_analysis(sim_dir, show=True, plots_dir=plots_dir)


if __name__ == '__main__':
    main()
