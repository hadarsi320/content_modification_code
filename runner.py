import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from multiprocessing import cpu_count
from os.path import exists
from time import sleep

from deprecated import deprecated

from gen_utils import run_bash_command, list_multiprocessing
from utils import parse_qrid, ensure_dirs, get_query_ids, load_word_embedding_model


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


def log_error(error_file, command):
    ensure_dirs(error_file)
    command = ' '.join([argument for argument in command.split()
                        if not ('word2vec_dump' in argument or 'output_dir' in argument)])
    with open(error_file, 'a') as f:
        f.write(f'{command}\n')


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
                      f' --output_dir={output_dir} --word2vec_dump={pickle_file}'
            print(f'{iteration}. Running command: {command}')
            try:
                run_bash_command(command)
            except Exception as e:
                print(f'#### Error occured in competition {qid} {", ".join(pid_list)}: \n{str(e)}\n')
                log_error(error_file, command)


def rerun_errors_2of2(output_dir, pickle_file):
    error_file = output_dir + 'error_file.txt'
    args = []
    with open(error_file) as f:
        for line in f:
            qid = line.split()[2]
            competitors = line.split()[5]
            args.append((qid, competitors))

    iteration = 0
    for qid, player_ids in args:
        iteration += 1
        command = f'python main.py --mode=2of2 --qid={qid} --bots={player_ids} --output_dir={output_dir}' \
                  f' --word2vec_dump={pickle_file}'
        print(f'{iteration}. Running command: {command}')
        try:
            run_bash_command(command)
            remove_from_error_file(error_file, qid, player_ids)
        except Exception as e:
            print(f'#### Error occured in competition {qid} {player_ids}: \n{str(e)}\n')


@deprecated()
def runner_2of5(output_dir, pickle_file, positions_file='./data/paper_data/documents.positions'):
    error_file = output_dir + 'error_file.txt'
    qid_list = sorted(get_query_ids(positions_file))
    iteration = 0
    for dummy_bot in [1, 2]:
        for qid in qid_list:
            iteration += 1

            if exists(output_dir + 'trec_files/trec_file_{}_{}'.format(qid, dummy_bot)):
                print('Competition qid={} dummy_bot={} has already been ran'.format(qid, dummy_bot))
                continue

            command = f'python main.py --mode=2of5 --qid={qid} --dummy_bot={dummy_bot}' \
                      f' --output_dir={output_dir}  --word2vec_dump={pickle_file}'
            print(f'{iteration}. Running command: {command}')
            try:
                run_bash_command(command)
            except Exception as e:
                print(f'#### Error occured in competition {qid} {dummy_bot}:\n{str(e)}\n')
                log_error(error_file, command)


@deprecated()
def runner_3of5(output_dir, pickle_file, positions_file='./data/paper_data/documents.positions'):
    error_file = output_dir + 'error_file.txt'
    qid_list = sorted(get_query_ids(positions_file))
    iteration = 0
    for qid in qid_list:
        iteration += 1

        if exists(output_dir + 'trec_files/trec_file_{}_both'.format(qid)):
            print('Competition qid={} has already been ran'.format(qid))
            continue

        command = f'python main.py --mode=3of5 --qid={qid} --output_dir={output_dir}  --word2vec_dump={pickle_file}'
        print(f'{iteration}. Running command: {command}')
        try:
            run_bash_command(command)

        except Exception as e:
            print(f'#### Error occured in competition {qid}:\n{str(e)}\n')
            log_error(error_file, command)


def runner_xof5(output_dir, results_dir, pickle_file, num_of_bots, top_refinement, print_interval, **kwargs):
    error_file = output_dir + 'error_file.txt'

    bots_list = defaultdict(list)
    if 'positions_file' in kwargs:
        mode = 'paper'
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
        mode = 'raifer'
        trec_file = kwargs.pop('trec_file')
        qid_list = sorted(get_query_ids(trec_file))
        for qid in qid_list:
            competitors = get_competitors_dict(trec_file)
            bots_list[qid] = list(combinations(competitors[qid], num_of_bots))
    else:
        raise ValueError('No file given to get bots')

    iteration = 0
    while len(qid_list) > 0:
        remove_list = []
        for qid in qid_list:
            iteration += 1
            if len(bots_list[qid]) > 0:
                bots = bots_list[qid].pop(0)
                if exists(output_dir + f'trec_files/similarity_{qid}_{",".join(bots)}.txt'):
                    print(f'{iteration}. Competition qid={qid} competitors={", ".join(bots)} has already been ran')
                    continue

                command = f'python main.py --output_dir={output_dir}' \
                          f' --mode={mode} ' \
                          f' --qid={qid} --bots={",".join(bots)}  --word2vec_dump={pickle_file}'
                if top_refinement is not None:
                    command += f' --top_refinement={top_refinement}'

                if iteration % print_interval == 0:
                    print(f'{iteration}. Running command: {command}')

                try:
                    run_bash_command(command)
                except Exception as e:
                    print(f'#### Error occured in competition {qid} {", ".join(bots)}: \n{str(e)}\n')
                    log_error(error_file, command)

                command = f'cp -r {output_dir}/trec_files {output_dir}/trectext_files {output_dir}/error_file.txt ' \
                          f'{results_dir}'
                run_bash_command(command)
            else:
                remove_list.append(qid)

        for qid in remove_list:
            qid_list.remove(qid)


def main(mode, source, print_interval=1, name=None, *args, **kwargs):
    positions_file_paper = './data/paper_data/documents.positions'
    trec_file_raifer = 'data/trec_file_original_sorted.txt'
    embedding_model_file = '/lv_local/home/hadarsi/work_files/word2vec_model/word2vec_model'

    if mode not in ['2of2', 'rerun_2of2'] + [f'{x}of5' for x in range(1, 6)]:
        raise ValueError(f'Illegal mode given {mode}')

    if mode.startswith('rerun'):
        print('Implement this rerunning thing')
        return

    if name is not None:
        results_dir = 'results/{}_{}/'.format(mode + datetime.now().strftime('_%m_%d_%H'), name)
        output_dir = 'output/{}_{}/'.format(mode + datetime.now().strftime('_%m_%d_%H'), name)
    else:
        results_dir = 'results/{}/'.format(mode + datetime.now().strftime('_%m_%d_%H'))
        output_dir = 'output/{}/'.format(mode + datetime.now().strftime('_%m_%d_%H'))

    ensure_dirs(results_dir, output_dir)

    word_embedding_model = load_word_embedding_model(embedding_model_file)
    word2vec_pkl = output_dir + 'word_embedding_model.pkl'
    with open(word2vec_pkl, 'wb') as f:
        pickle.dump(word_embedding_model, f)

    if mode == '2of2':
        runnner_2of2(output_dir, word2vec_pkl)

    elif mode.endswith('of5'):
        num_of_bots = int(mode[0])
        if source == 'paper':
            runner_xof5(output_dir, results_dir, word2vec_pkl, num_of_bots, args[0], print_interval,
                        positions_file=positions_file_paper, **kwargs)
        elif source == 'raifer':
            runner_xof5(output_dir, results_dir, word2vec_pkl, num_of_bots, args[0], print_interval,
                        trec_file=trec_file_raifer, **kwargs)

    os.remove(word2vec_pkl)


if __name__ == '__main__':
    # mode = sys.argv[1]
    # source = sys.argv[2]
    # if len(sys.argv) > 3:
    #     main(mode, source, name=sys.argv[3])
    # else:
    #     main(mode, source)

    workers = cpu_count() - 1
    modes = [f'{x}of5' for x in range(1, 6)]
    top_refinement_methods = ['past_top', 'highest_rated_inferiors', None, 'acceleration']

    params_list = [(mode, 'raifer', 25, top_ref, top_ref) for top_ref in top_refinement_methods for mode in modes]
    list_multiprocessing(params_list, main, workers=workers)
