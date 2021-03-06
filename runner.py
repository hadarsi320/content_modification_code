import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from multiprocessing import Pool
from os.path import exists

from utils import constants
from utils.bash_utils import run_bash_command
from competition_main import competition_setup
from utils.general_utils import parse_qrid, ensure_dirs, get_query_ids, load_word_embedding_model


def get_competitors_dict(trec_file: str):
    competitors_dict = defaultdict(list)
    with open(trec_file) as f:
        for line in f:
            qrid = line.split()[0]
            epoch, qid = parse_qrid(qrid)
            if epoch != '01':
                continue
            competitor = line.split()[2].split('-')[-1]
            competitors_dict[qid].append(competitor)
    for qid in competitors_dict:
        competitors_dict[qid].sort()
    return dict(competitors_dict)


def log_error(error_dir, command, error):
    ensure_dirs(error_dir)
    command = ' '.join([argument for argument in command.split()
                        if not ('word2vec_dump' in argument or 'output_dir' in argument)])
    with open(error_dir + command, 'w') as f:
        f.write(str(error))


def remove_from_error_file(error_file, qid, players):
    lines = []
    with open(error_file, 'r') as f:
        for line in f:
            last_qid = line.split()[2]
            last_players = line.split()[5]
            if last_qid != qid or last_players != players:
                lines.append(line)

    with open(error_file, 'w') as f:
        for line in lines:
            f.write(line)


def runnner_2of2(output_dir, pickle_file, trec_file='./data/trec_file_original_sorted.txt'):
    error_file = output_dir + 'error_file.txt'
    competitors = get_competitors_dict(trec_file)
    competitors_combinations = {qid: list(combinations(competitors[qid], 2)) for qid in competitors}
    iteration = 0
    for i in range(10):
        for qid in competitors_combinations:
            iteration += 1
            pid_list = sorted(competitors_combinations[qid][i])
            if exists(output_dir + 'similarity_results/similarity_{}_{}.txt'.format(qid, ','.join(pid_list))):
                print('Competition qid={} competitors={} has already been ran'.format(qid, ', '.join(pid_list)))
                continue

            command = f'python main.py --mode=2of2 --qid={qid} --bots={",".join(pid_list)}' \
                      f' --output_dir={output_dir}'
            print(f'{iteration}. Running command: {command}')
            try:
                run_bash_command(command)
            except Exception as e:
                print(f'#### Error occured in competition {qid} {", ".join(pid_list)}: \n{str(e)}\n')
                log_error(error_file, command, e)


def get_bots(num_of_bots, total_players, **kwargs):
    bots_list = {}
    if 'positions_file' in kwargs:
        positions_file = kwargs.pop('positions_file')
        qid_list = sorted(get_query_ids(positions_file))
        for qid in qid_list:
            if num_of_bots == 1:
                bots_list[qid] = [['BOT']]
            elif num_of_bots == 2:
                bots_list[qid] = [['BOT', 'DUMMY1'], ['BOT', 'DUMMY2']]
            elif num_of_bots == 3:
                bots_list[qid] = [['BOT', 'DUMMY1', 'DUMMY2']]

    elif 'trec_file' in kwargs:
        trec_file = kwargs.pop('trec_file')
        competitors = get_competitors_dict(trec_file)
        qid_list = sorted(get_query_ids(trec_file))
        for qid in qid_list:
            if len(competitors[qid]) == total_players:
                bots_list[qid] = list(combinations(competitors[qid], num_of_bots))
    else:
        raise ValueError('No source file given')

    return bots_list


def run_all_queries(output_dir, results_dir, num_of_bots, tr_method, validation_method, word2vec_pickle,
                    print_interval=5, total_players=5, **kwargs):
    error_dir = output_dir + 'errors/'
    bots_list = get_bots(num_of_bots, total_players, **kwargs)
    mode = kwargs.pop('mode')

    iteration = 1
    for qid in bots_list:
        for bots in bots_list[qid]:
            run_description = f'output_dir={output_dir} qid={qid} bots={",".join(bots)} trm={tr_method} ' \
                              f'val={validation_method}'
            if iteration == 1 or iteration % print_interval == 0:
                print(f'{iteration}. {run_description}')

            stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            try:
                competition_setup(mode=mode, output_dir=output_dir, qid=qid, bots=bots, word2vec_pickle=word2vec_pickle,
                                  top_refinement=tr_method, validation_method=validation_method, mute=True)
                sys.stdout = stdout
            except Exception as e:
                sys.stdout = stdout
                print(f'#### Error occured in competition {qid} {", ".join(bots)}: \n{str(e)}\n')
                log_error(error_dir, run_description, e)

            ensure_dirs(results_dir)
            for directory in ['trec_files', 'trectext_filces', 'errors', 'replacements']:
                if os.path.exists(f'{output_dir}/{directory}'):
                    command = f'cp -r {output_dir}/{directory} {results_dir}'
                    run_bash_command(command)
            iteration += 1


def run_all_competitions(mode, tr_method, validation_method, run_name,
                         source='goren', positions_file_paper='./data/paper_data/documents.positions',
                         trec_file_raifer='data/trec_file_original_sorted.txt',
                         embedding_model_file='/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model'):
    """
    A function which runs all possible queries and bot combinations for the given arguments
    top_refinement: one of the following options: 'vanilla', 'acceleration', 'past_top', 'highest_rated_inferiors',
                              'past_targets', 'everything'
    """
    if mode.startswith('rerun'):
        print('Implement this rerunning thing')
        return

    name = tr_method + '_' + validation_method
    if run_name is not '':
        name += '_' + run_name

    folder_name = '_'.join((mode, datetime.now().strftime('%m_%d'), name))
    results_dir = f'results/{folder_name}/'
    output_dir = f'output/{folder_name}/'

    print(f'Running mode {mode} with top refinement method {tr_method} and validation method {validation_method}')

    word2vec_pkl = output_dir + 'word_embedding_model.pkl'
    ensure_dirs(output_dir)
    word_embedding_model = load_word_embedding_model(embedding_model_file)
    with open(word2vec_pkl, 'wb') as f:
        pickle.dump(word_embedding_model, f)

    assert mode.endswith('of5')
    num_of_bots = int(mode[0])
    if source == 'goren':
        kwargs = {'mode': source, 'positions_file': positions_file_paper}
    elif source == 'raifer':
        kwargs = {'mode': source, 'trec_file': trec_file_raifer}
    else:
        raise ValueError(f'Illegal source given {source}')

    run_all_queries(output_dir, results_dir, num_of_bots, tr_method, validation_method, word2vec_pkl, **kwargs)
    print(f'\t\tFinished running {name} with mode {mode} and TRM {tr_method} and val method {validation_method}')
    os.remove(word2vec_pkl)


def main():
    run_name = input('Insert run name\n')

    results_dir = 'results/'
    modes = ['1of5']
    tr_methods = [constants.VANILLA, constants.ACCELERATION, constants.HIGHEST_RATED_INFERIORS]
    validation_methods = [constants.NAIVE, constants.PROBABILITIES, constants.PREDICTION]

    args = []
    for mode in modes:
        for tr_method in tr_methods:
            for validation_method in validation_methods:
                folder_name = f'{tr_method}_{validation_method}'
                if len(run_name) > 0:
                    folder_name += '_' + run_name
                reg_exp = mode + '.*' + '_' + folder_name
                if all(re.match(reg_exp, file) is None for file in os.listdir(results_dir)):
                    args.append((mode, tr_method, validation_method, run_name))

    with Pool() as p:
        p.starmap(run_all_competitions, args)


if __name__ == '__main__':
    main()
