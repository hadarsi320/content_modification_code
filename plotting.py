import os
import sys
from collections import defaultdict, Counter
from os import listdir

import numpy as np
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm

import competition_main
from bot_competition import generate_document_tfidf_files
from data_analysis import compute_average_rank, compute_average_promotion, cumpute_atd
from utils import read_competition_trec_file, normalize_dict_len, ensure_dirs, read_positions_file, read_trec_dir, \
    get_competitors, create_index, create_documents_workingset, get_num_rounds, read_raw_trec_file, read_trec_file, \
    format_name
from vector_functionality import document_tfidf_similarity

COLORS = {'green': '#32a852', 'red': '#de1620', 'blue': '#1669de', 'orange': '#f28e02', 'purple': '#8202f2'}


def item_counts(l: list, normalize=False):
    counts = np.array([l.count(value) for value in sorted(set(l))])
    if normalize:
        return counts / np.sum(counts)
    return counts


def plot(data, start=0, stop=None, shape='o-', title=None, x_label=None, y_label=None, savefig=None, show=False,
         fig_size=(6, 4)):
    # plt.close()
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
    if savefig:
        plt.savefig(savefig)
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
    max_val = max(max(value[1]) for key in results for value in results[key])
    min_val = min(min(value[1]) for key in results for value in results[key])

    max_tick = (max_val // jump_len + 4) * jump_len
    min_tick = min_val // jump_len * jump_len

    return np.arange(min_tick, max_tick, jump_len)


def compare_competitions(title, show=True, plots_dir='plots/', legend_ncols=2, **kwargs):
    ranked_lists_dict = {}
    competitors_lists_dict = {}

    positions_files = kwargs.pop('positions_files', [])
    for key in positions_files:
        ranked_lists_dict[key] = read_positions_file(positions_files[key])
        competitors_lists_dict[key] = {qid: next(iter(ranked_lists_dict[key][qid].values()))
                                       for qid in ranked_lists_dict[key]}

    trec_dirs = kwargs.pop('trec_dirs', [])
    for key in trec_dirs:
        ranked_lists_dict[key], competitors_lists_dict[key] = read_trec_dir(trec_dirs[key])

    # Normalizing is a bad practice, there should be no errors
    # rounds = []
    # for key in ranked_lists_dict:
    #     rounds.append(normalize_dict_len(ranked_lists_dict[key]))
    # assert all(num_rounds == max(rounds) for num_rounds in rounds)
    # num_rounds = max(rounds)

    rounds = max(len(ranked_lists_dict[key][name]) for key in ranked_lists_dict for name in ranked_lists_dict[key])
    assert all(
        len(ranked_lists_dict[key][name]) == rounds for key in ranked_lists_dict for name in ranked_lists_dict[key])

    groups = ['students', 'bots']
    colors = [COLORS[color] for color in ['blue', 'red', 'green', 'orange', 'purple']]
    shapes_list = {'bots': 'D', 'students': 'o'}
    y_labels = ['Average Rank', 'Raw Rank Promotion', 'Scaled Rank Promotion']
    x_ticks_list = [range(1, rounds + 1), range(2, rounds + 1), range(2, rounds + 1)]
    jump_lengths = [0.25, 0.1, 0.05]
    functions = [lambda x, y, z, w: compute_average_rank(x, y, z, is_paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=False, is_paper_data=w),
                 lambda x, y, z, w: compute_average_promotion(x, y, z, scaled=True, is_paper_data=w)]

    if 'axs' in kwargs:
        axs = kwargs.pop('axs')
    else:
        figsize = kwargs.pop('figsize', (27, 8))
        fig, axs = plt.subplots(ncols=3, figsize=figsize)

        if 'suptitle' in kwargs:
            fig.suptitle(kwargs.pop('suptitle'), fontsize=18)
    plt.rcParams.update({'font.size': 12})

    for i in range(3):
        axis = axs[i]
        x_ticks = x_ticks_list[i]
        function = functions[i]
        y_label = y_labels[i]

        results = {group: [] for group in groups}
        for j, key in enumerate(ranked_lists_dict):
            ranked_lists = ranked_lists_dict[key]
            competitors_lists = competitors_lists_dict[key]
            is_paper_data = key in positions_files
            for group in groups:
                results[group].append((key, function(ranked_lists, competitors_lists, group, is_paper_data), colors[j]))

        for group in groups:
            for key, result, color in results[group]:
                label = f'{key}- {group}'
                shape = shapes_list[group]

                assert len(x_ticks) == len(result)
                axis.plot(x_ticks, result, label=label, color=color, marker=shape, markersize=10)

        axis.set_title(title)
        axis.set_ylabel(y_label)
        axis.legend(ncol=legend_ncols, loc='upper right')
        axis.set_xticks(x_ticks)
        axis.set_yticks(get_optimal_yticks(results, jump_len=jump_lengths[i]))
        axis.set_xlabel('Round')

    if 'savefig' in kwargs:
        plt.savefig(plots_dir + kwargs['savefig'])
    if show and 'axs' not in kwargs:
        plt.show()


def compare_trm_atd(trec_dirs_dict, show=True, **kwargs):
    """
    trm - Top Refinement Methods
    atd - Average Top Duration
    """
    colors = [COLORS[color] for color in ['blue', 'red', 'green', 'orange', 'purple']]

    plt.figure().clear()
    for i, tr_method in enumerate(trec_dirs_dict):
        color = colors[i]
        trec_dirs = trec_dirs_dict[tr_method]

        list_of_ranked_lists = {}
        competitions = sorted(trec_dirs)
        for key in competitions:
            list_of_ranked_lists[key], _ = read_trec_dir(trec_dirs[key])
        competition_atd = [cumpute_atd(list_of_ranked_lists[key])
                           for key in list_of_ranked_lists]
        students_atd = [value[0] for value in competition_atd[:-1]]
        bots_atd = [value[1] for value in competition_atd]
        plt.plot(competitions[:-1], students_atd, label=tr_method + ': students', color=color, marker='o')
        plt.plot(competitions, bots_atd, label=tr_method + ': bots', color=color, marker='D')
    plt.xlabel('Competition Type')
    plt.title('Average First Rank Duration')
    plt.legend()

    if show:
        plt.show()
    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'])


def compare_to_paper_data():
    compare_competitions('paper data', show=False, legend_ncols=1, figsize=(15, 8),
                         trec_dirs={'Recreated Competitions': 'results/1of5_10_27_11_rerunning_paper/trec_files'},
                         positions_files={'Paper Competitions': 'data/paper_data/documents.positions'},
                         suptitle='Comparison of Bot to the Paper Plots',
                         savefig='Comparison to Paper Plots')


def plot_rank_distribution(competition_dir, position, show=True, set_ylabel=False, **kwargs):
    ALPHA = 1
    ranked_lists, competitors_dict = read_trec_dir(competition_dir+'/trec_files/')
    rounds = len(next(iter(ranked_lists.values())))
    max_rank = len(next(iter(competitors_dict.values())))

    bot_ranks = defaultdict(list)
    # student_ranks = defaultdict(list)

    for competition in ranked_lists:
        bots = competition.split('_')[3].split(',')
        for epoch in ranked_lists[competition]:
            for i, pid in enumerate(ranked_lists[competition][epoch]):
                if pid in bots:
                    bot_ranks[epoch].append(i)
                # else:
                #     student_ranks[epoch].append(i)

    if 'axes' in kwargs:
        axes = kwargs.pop('axes')
        assert len(axes) == rounds
        format_plot = False
    else:
        _, axes = plt.subplots(nrows=rounds, figsize=(8, 3 * rounds), sharey='col')
        format_plot = True

    if 'method' in kwargs:
        method = kwargs.pop('method')
        labels = [f'{method}: Students', f'{method}: Bots']
    else:
        labels = ['Students', 'Bots']

    bots_kwargs = {}
    if 'colors' in kwargs:
        bots_kwargs['color'] = kwargs.pop('color')

    x_axis = np.arange(1, max_rank+1)
    width = 0.4
    for epoch, axis in zip(bot_ranks, axes):
        if position == 'left':
            axis.bar(x_axis-width/2, item_counts(bot_ranks[epoch], normalize=True),
                     width=width, alpha=ALPHA, label=labels[1], **bots_kwargs)
        else:
            axis.bar(x_axis+width/2, item_counts(bot_ranks[epoch], normalize=True),
                     width=width, alpha=ALPHA, label=labels[1], **bots_kwargs)

        if format_plot:
            axis.legend()
            axis.set_title(f'Round {epoch}')
            axis.set_xlabel('Rank')
            if set_ylabel:
                axis.set_ylabel('Rank Distribution')

    if 'savefig' in kwargs:
        plt.savefig(kwargs['savefig'])

    if show is True:
        plt.show()


def plot_top_distribution(trec_dir, show=True, set_ylabel=True, **kwargs):
    ranked_lists, competitors_dict = read_trec_dir(trec_dir)
    rounds = len(next(iter(ranked_lists.values())))
    total_competitions = len(ranked_lists)

    bots_top = Counter()
    students_top = Counter()

    for competition in ranked_lists:
        bots = competition.split('_')[3].split(',')
        for epoch in ranked_lists[competition]:
            top_pid = ranked_lists[competition][epoch][0]
            if top_pid in bots:
                bots_top[epoch] += 1
            else:
                students_top[epoch] += 1

    results = {epoch:
                   [students_top[epoch] / total_competitions, bots_top[epoch] / total_competitions]
               for epoch in students_top}

    # results = {'students': [students_top[epoch] / total_competitions for epoch in students_top],
    #            'bots': [bots_top[epoch] / total_competitions for epoch in bots_top]}

    if 'axes' in kwargs:
        axes = kwargs.pop('axes')
        assert len(axes) == rounds
    else:
        _, axes = plt.subplots(nrows=rounds, figsize=(5, 3 * rounds), sharey='col')

    x_ticks = [0, 1]
    for axis, epoch in zip(axes, sorted(results)):
        axis.bar(x_ticks, results[epoch])
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(['Students', 'Bots'])
        axis.set_title(f'Round {epoch}')
        if set_ylabel:
            axis.set_ylabel('Distribution')

    if show:
        plt.show()


def plot_similarity_to_winner(comp_dirs_dict: dict, rounds: int, show=True, **kwargs):
    tmp_dir = 'plotting_tmp/'
    index = tmp_dir + 'index'
    doc_ws_file = tmp_dir + 'doc_ws_file.txt'
    tfidf_dir = tmp_dir + 'tfidf/'

    top_similarity = {method: defaultdict(list) for method in comp_dirs_dict}
    for method, comp_dir in comp_dirs_dict.items():
        trectext_dir = comp_dir + '/trectext_files/'

        for file in tqdm(os.listdir(trectext_dir)):
            qid = file.split('_')[1]
            bots = file.split('_')[2].split('.')[0].split(',')
            trec_file = f'{comp_dir}/trec_files/trec_file_{qid}_{",".join(bots)}'
            competitors = get_competitors(trec_file)
            ranked_list = read_trec_file(trec_file)

            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            create_index(trectext_dir+file, new_index_name=index, indri_path=competition_main.indri_path)
            create_documents_workingset(doc_ws_file, competitors, qid, total_rounds=rounds)
            generate_document_tfidf_files(doc_ws_file, output_dir=tfidf_dir,
                                          swig_path=competition_main.swig_path,
                                          base_index=competition_main.clueweb_index, new_index=index)
            sys.stdout = stdout

            for epoch in ranked_list:
                top_document = tfidf_dir + ranked_list[epoch][qid][0]
                for bot in bots:
                    bot_document = tfidf_dir + f'ROUND-{epoch}-{qid}-{bot}'
                    top_similarity[method][epoch].append(document_tfidf_similarity(top_document, bot_document))

            shutil.rmtree(tmp_dir)
    
    results = {method:
                   {epoch:
                        np.mean(top_similarity[method][epoch])
                    for epoch in top_similarity[method]}
               for method in top_similarity}
    for method in results:
        plt.plot(range(1, rounds+1), results[method].values(), label=format_name(method))
    plt.legend()
    plt.title('Similarity of Bot Documents to Winning Document')
    plt.xlabel('Round')
    plt.ylabel('Average Similarity')

    if 'savefig' in kwargs:
        plt.savefig(kwargs.pop('savefig'))
    if show:
        plt.show()


def plot_trm_comparisons(modes, tr_methods, performance_comparison=False, average_top_duration=False,
                         rank_distribution=False, top_distribution=False, similarity_to_winner=False,
                         **kwargs):
    plots_dir = './plots'
    results_dir = 'results/'
    ensure_dirs(plots_dir)
    rounds = kwargs.pop('rounds', None)

    comp_dirs = defaultdict(dict)
    # TODO update usage of comp_dirs to fit the new transfer from trec dirs to comp dirs
    for mode in modes:
        for method in tr_methods:
            competitions = [competition for competition in os.listdir(results_dir)
                            if mode in competition and method in competition]
            latest = sorted(competitions)[-1]
            comp_dirs[mode][method] = results_dir + latest

    if performance_comparison:
        competitions_list = list(comp_dirs.values())[:-1]
        labels = [f'{i + 1} bots out of 5' for i in range(5)]

        _, axes_mat = plt.subplots(ncols=3, nrows=len(competitions_list), figsize=(30, 10 * len(competitions_list)),
                                   squeeze=False)
        for i, axes in enumerate(axes_mat):
            competitions = competitions_list[i]
            compare_competitions(trec_dirs=competitions, axs=axes, title=labels[i], show=False)
        plt.savefig(plots_dir + '/Comparison of Top Refinement Methods HRI')

    if average_top_duration:
        competitions_list = list(comp_dirs.values())
        competitions_list_rev = {method:
                                     {f'{x + 1}of5': competitions_list[x][method] for x in
                                      range(len(competitions_list))}
                                 for method in tr_methods}

        compare_trm_atd(competitions_list_rev, show=False,
                        savefig=plots_dir + '/Average First Place Duration ' + ' vs '.join(tr_methods))

    if rank_distribution:
        assert len(tr_methods) == 2
        for mode in modes:
            fig, axes = plt.subplots(nrows=rounds, figsize=(10, 3 * rounds), squeeze=True, sharey='none')
            for i, method in enumerate(tr_methods):
                competition_dir = comp_dirs[mode][method]
                pos = 'left' if i == 0 else 'right'
                plot_rank_distribution(competition_dir, pos, axes=axes, show=False, set_ylabel=i == 0, method=method)

            fig.suptitle('Rank Distribution of Students and Bots', fontsize=18)
            for i, axis in enumerate(axes):
                axis.legend()
                axis.set_title(f'Round {i+1}')
                axis.set_xlabel('Rank')
                axis.set_ylabel('Rank Distribution')
            plt.tight_layout(rect=(0, 0.03, 1, 0.97))

            plt.savefig(plots_dir + '/Rank Distribution ' + ' vs '.join(tr_methods))
            plt.close()
            # plt.show()

    if top_distribution:
        for mode in modes:
            fig, axes_mat = plt.subplots(ncols=len(tr_methods), nrows=rounds,
                                         figsize=(10, 3 * rounds), squeeze=False, sharey='row')
            for i, (method, axes) in enumerate(zip(tr_methods, axes_mat.transpose())):
                plot_top_distribution(comp_dirs[mode][method], axes=axes, show=False, set_ylabel=i == 0)
            fig.suptitle('Top Distribution for TR methods: {}'.format(', '.join(tr_methods)), fontsize=18)
            plt.tight_layout(rect=(0, 0.03, 1, 0.97))
            plt.savefig(plots_dir+'/Top Distribution ' + ' vs '.join(tr_methods))
            plt.show()
            plt.close()

    if similarity_to_winner:
        for mode in comp_dirs:
            plot_similarity_to_winner(comp_dirs[mode], rounds, show=False,
                                      savefig=plots_dir+'/Similarity To Winner Document ' + ' vs '.join(tr_methods))
            plt.close()


def main():
    modes = ['1of5']  # [f'{x + 1}of5' for x in range(5)]
    tr_methods = ['vanilla', 'past_targets']
    plot_trm_comparisons(modes, tr_methods, rank_distribution=True, top_distribution=False, similarity_to_winner=True,
                         rounds=8)
    # compare_to_paper_data()


if __name__ == '__main__':
    main()
