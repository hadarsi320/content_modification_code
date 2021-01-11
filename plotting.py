import os
import re
from collections import defaultdict, Counter
from os import listdir
from typing import Dict

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import utils.general_utils as utils
from utils.data_statistics import compute_average_rank, compute_average_promotion, cumpute_atd
from utils.readers import TrecReader
# from utils import read_competition_trec_file, normalize_dict_len, ensure_dirs, read_positions_file, read_trec_dir, \
#     get_competitors, read_features_dir, parse_doc_id, \
#     get_next_epoch

COLORS = {'green': '#32a852', 'red': '#de1620', 'blue': '#1669de', 'orange': '#f28e02', 'purple': '#8202f2',
          'sky': '#0acef5'}


def get_rank_distribution(rank_list: list, total_ranks):
    assert all(value in range(total_ranks) for value in rank_list)
    counts = np.array([rank_list.count(value) for value in range(total_ranks)])
    return counts / np.sum(counts)


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

    comp_dirs = kwargs.pop('comp_dirs', [])
    for key in comp_dirs:
        ranked_lists_dict[key], competitors_lists_dict[key] = read_trec_dir(comp_dirs[key] + '/trec_files/')

    rounds = max(len(ranked_lists_dict[key][name]) for key in ranked_lists_dict for name in ranked_lists_dict[key])
    assert all(
        len(ranked_lists_dict[key][name]) == rounds for key in ranked_lists_dict for name in ranked_lists_dict[key])

    groups = ['students', 'bots']
    colors = [COLORS[color] for color in ['blue', 'red', 'green', 'orange', 'purple', 'sky']]
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
    # ranked_lists, competitors_dict = read_trec_dir(competition_dir + '/trec_files/')
    # rounds = len(next(iter(ranked_lists.values())))
    # max_rank = len(next(iter(competitors_dict.values())))

    comp_trec_reader = TrecReader(trec_dir=f'{competition_dir}/trec_files/')
    rounds = comp_trec_reader.num_epochs()
    max_rank = comp_trec_reader.max_rank()

    bot_ranks = defaultdict(list)
    for competition in comp_trec_reader.queries():
        bots = competition.split('_')[1].split(',')
        for epoch in comp_trec_reader.epochs():
            for i, doc_id in enumerate(comp_trec_reader[epoch][competition]):
                pid = utils.parse_doc_id(doc_id)[2]
                if pid in bots:
                    bot_ranks[epoch].append(i)

    if 'axes' in kwargs:
        axes = kwargs.pop('axes')
        # assert len(axes) == rounds
        format_plot = False
    else:
        _, axes = plt.subplots(nrows=rounds, figsize=(8, 3 * rounds), sharey='col')
        format_plot = True

    if 'method' in kwargs:
        label = kwargs.pop('method')
    else:
        label = 'Bots'

    bots_kwargs = {}
    if 'colors' in kwargs:
        bots_kwargs['color'] = kwargs.pop('color')

    x_axis = np.arange(1, max_rank + 1)
    width = 0.4
    for epoch, axis in zip(bot_ranks, axes):
        res = get_rank_distribution(bot_ranks[epoch], total_ranks=max_rank)
        if position == 'left':
            axis.bar(x_axis - width / 2, res,
                     width=width, alpha=ALPHA, label=label, **bots_kwargs)
        else:
            axis.bar(x_axis + width / 2, res,
                     width=width, alpha=ALPHA, label=label, **bots_kwargs)

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


def compare_rank_distributions(comp_dirs, rounds, run_name, plots_dir):
    assert len(comp_dirs) == 2

    fig, axes = plt.subplots(nrows=rounds, figsize=(10, 3 * rounds), squeeze=True, sharey='none')
    plt.tight_layout(pad=4)
    for i, method in enumerate(comp_dirs):
        competition_dir = comp_dirs[method]
        pos = 'left' if i == 0 else 'right'
        plot_rank_distribution(competition_dir, pos, axes=axes, show=False, set_ylabel=i == 0, method=method)

    fig.suptitle(f'Rank Distribution of Students and Bots {run_name}', fontsize=18, y=.995)
    for i, axis in enumerate(axes):
        axis.legend()
        axis.set_title(f'Round {i + 1}')
        axis.set_xlabel('Rank')
        axis.set_ylabel('Rank Distribution')

    # plt.savefig(plots_dir + '/Rank Distribution ' + plot_name)
    plt.show()
    plt.close()


def separate_features_dict(features_dict, ranked_lists) -> Dict[str, np.ndarray]:
    res = defaultdict(list)
    for comp_id in features_dict:
        for doc_id in features_dict[comp_id]:
            epoch, _, pid = utils.parse_doc_id(doc_id)
            rank = ranked_lists[comp_id][epoch].index(pid)
            next_epoch = utils.get_next_epoch(epoch)
            next_rank = ranked_lists[comp_id][next_epoch].index(pid)
            if rank == 0:
                if next_rank <= rank:
                    res['Top success'].append(features_dict[comp_id][doc_id])
                else:
                    res['Top fail'].append(features_dict[comp_id][doc_id])
            else:
                res['General'].append(features_dict[comp_id][doc_id])

    res = {key: np.array(res[key]) for key in res}
    return res


def plot_feature_values(competition_dir, name, show=False, seperate=False, ttest=False, alpha=0.05):
    feature_names = ["FractionOfQueryWordsIn", "FractionOfQueryWordsOut", "CosineToCentroidIn", "CosineToCentroidInVec",
                     "CosineToCentroidOut", "CosineToCentroidOutVec", "CosineToWinnerCentroidInVec",
                     "CosineToWinnerCentroidOutVec", "CosineToWinnerCentroidIn", "CosineToWinnerCentroidOut",
                     "SimilarityToPrev", "SimilarityToRefSentence", "SimilarityToPred", "SimilarityToPrevRef"]

    ranked_lists, competitors_dict = read_trec_dir(competition_dir + '/trec_files/')
    features_dict = read_features_dir(f'{competition_dir}/replacements')
    stat_indices = []

    if seperate:
        features = separate_features_dict(features_dict, ranked_lists)

        if ttest:
            for i, feature_name in enumerate(feature_names):
                features_success = [vec[i] for vec in features['Top success']]
                features_fail = [vec[i] for vec in features['Top fail']]
                statistic, pvalue = scipy.stats.ttest_ind(a=features_success, b=features_fail, equal_var=False)
                if pvalue <= alpha:
                    print(f'T Test on {feature_name:>30}: statistic value {statistic:.3f} pvalue {pvalue:.3f}')
                    stat_indices.append(i)

            # for key in features:
            #     features[key] = features[key][:, stat_indices]
            features = {key: features[key][:, stat_indices] for key in features}
            feature_names = [feature_names[i] for i in stat_indices]

    else:
        features = {'': [features_dict[comp][doc] for comp in features_dict for doc in features_dict[comp]]}

    if show:
        plt.figure(figsize=(12, 5))
        plt.subplots_adjust(bottom=0.5)

    for key in features:
        plt.scatter(x=np.arange(features[key].shape[1]), y=(np.average(features[key], axis=0)),
                    label=f'{name} {key}'.strip(), marker='_', linewidths=3)
    plt.xticks(range(len(feature_names)), feature_names, rotation='vertical')

    if show:
        if ttest:
            plt.title(f'Statistically Different Feature Values {name}')
        else:
            plt.title(f'Feature Values {name}')
        plt.legend()
        plt.show()


def plot_trm_comparisons(modes, tr_methods, plot_name, run_name=None, performance_comparison=False,
                         average_top_duration=False, rank_distribution=False, similarity_to_winner=False,
                         feature_values=False, **kwargs):
    plots_dir = './plots'
    results_dir = 'results/'
    utils.ensure_dirs(plots_dir)
    rounds = kwargs.pop('rounds', None)

    comp_dirs = defaultdict(dict)
    # TODO update usage of comp_dirs to fit the new transfer from trec dirs to comp dirs
    for mode in modes:
        for method in tr_methods:
            reg_exp = mode + '.*' + method + (('_' + run_name) if run_name is not None else '') + '$'
            competitions = [comp_dir for comp_dir in os.listdir(results_dir)
                            if re.match(reg_exp, comp_dir)]
            latest = sorted(competitions)[-1]
            comp_dirs[mode][method] = results_dir + latest + '/'

    if performance_comparison:
        competitions_list = [comp_dirs[mode] for mode in comp_dirs if mode != '5of5']
        labels = [f'{i + 1} bots out of 5' for i in range(5)]

        _, axes_mat = plt.subplots(ncols=3, nrows=len(competitions_list), figsize=(30, 10 * len(competitions_list)),
                                   squeeze=False)
        for i, axes in enumerate(axes_mat):
            competitions = competitions_list[i]
            compare_competitions(comp_dirs=competitions, axs=axes, title=labels[i], show=False)
        # plt.savefig(plots_dir + '/Comparison of Top Refinement Methods ' + plot_name)
        plt.show()

    if average_top_duration:
        competitions_list = list(comp_dirs.values())
        competitions_list_rev = {method:
                                     {f'{x + 1}of5': competitions_list[x][method] for x in
                                      range(len(competitions_list))}
                                 for method in tr_methods}

        compare_trm_atd(competitions_list_rev, show=False,
                        savefig=plots_dir + '/Average First Place Duration ' + plot_name)

    if rank_distribution:
        assert len(tr_methods) == 2
        for mode in modes:
            compare_rank_distributions(comp_dirs[mode], rounds, run_name, plots_dir)

    if similarity_to_winner:
        for mode in comp_dirs:
            plot_similarity_to_winner(comp_dirs[mode], rounds, show=False,
                                      savefig=plots_dir + '/Similarity To Winner Document ' + plot_name)
            plt.close()

    if feature_values:
        for mode in modes:
            plt.figure(figsize=(12, 5))
            plt.subplots_adjust(bottom=0.5)
            for method in tr_methods:
                plot_feature_values(comp_dirs[mode][method], name=method, show=False, seperate=method != 'vanilla')
            plt.legend()
            plt.show()


def main():
    # modes = ['1of5']
    # runs = ['proba-classify', 'alteration_classifier', None]
    # tr_methods = ['acceleration', 'highest_rated_inferiors']
    # for run in runs:
    #     for method in tr_methods:
    #         plot_trm_comparisons(modes, ['vanilla', method], run_name=run, rounds=8, plot_name=None,
    #                              performance_comparison=False, rank_distribution=True, feature_values=False)

    # comp_dirs = {'Naive': 'results/1of5_10_22_22_acceleration/',
    #              'Classifier Predictions': 'results/1of5_12_21_acceleration_alteration_classifier/',
    #              'Classifier Probabilities': 'results/1of5_12_26_acceleration_proba-classify/'}
    comp_dirs = {'Naive': 'results/1of5_10_23_04_highest_rated_inferiors/',
                 'Classifier Predictions': 'results/1of5_12_21_highest_rated_inferiors_alteration_classifier/',
                 'Classifier Probabilities': 'results/1of5_12_26_highest_rated_inferiors_proba-classify/'}
    vanilla_comp_dir = 'results/1of5_10_22_13_vanilla/'

    fig, axes = plt.subplots(ncols=3, nrows=8, figsize=(15, 2.5 * 8), squeeze=True, sharey='row')
    plt.tight_layout(pad=4)
    for i, key in enumerate(comp_dirs):
        ax = axes[:, i]
        plot_rank_distribution(vanilla_comp_dir, 'left', axes=ax, show=False, set_ylabel=True, method='Vanilla')
        plot_rank_distribution(comp_dirs[key], 'right', axes=ax, show=False, set_ylabel=False, method='HRI')

    fig.suptitle(f'Rank Distribution of Students and Bots Using HRI', fontsize=18, y=.995)
    for i, row in enumerate(axes):
        for axis, key in zip(row, comp_dirs):
            axis.legend()
            axis.set_title(f'{key}- Round {i + 1}')
            axis.set_xlabel('Rank')
            axis.set_ylabel('Rank Distribution')

    # plt.savefig(plots_dir + '/Rank Distribution ' + plot_name)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
