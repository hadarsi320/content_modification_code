import os
import sys

from utils import parse_qrid, ensure_dir, xor
from collections import defaultdict
from itertools import combinations
from os.path import exists
from gen_utils import run_and_print, run_bash_command
from random import sample


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
    return dict(competitors_dict)


def log_error(error_file, qid, comptitors=None, dummy_bot=None):
    assert xor(comptitors, dummy_bot)
    ensure_dir(error_file)
    with open(error_file, 'a') as f:
        if comptitors:
            f.write(f'Error in {qid} with competitors {",".join(comptitors)}\n')
        else:
            f.write(f'Error in {qid} with dummy bot index {dummy_bot}\n')


def get_queries(positions_file):
    qid_list = []
    with open(positions_file) as f:
        for line in f:
            qid = line.split()[0]
            if qid not in qid_list:
                qid_list.append(qid)
    return qid_list


def runnner_2of2(output_dir, trec_file='./data/trec_file_original_sorted.txt'):
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

            command = f'python main.py --mode=2of2 --qid={qid} --competitors={",".join(pid_list)}' \
                      f' --output_dir={output_dir}'
            print(f'{iteration}. Running command: {command}')
            try:
                run_bash_command(command)
            except Exception as e:
                print(f'#### Error occured in competition {qid} {", ".join(pid_list)}: \n{str(e)}\n')
                log_error(error_file, qid, comptitors=pid_list)


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


def rerun_errors_2of2(output_dir):
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
        command = f'python main.py --mode=2of2 --qid={qid} --competitors={player_ids} --output_dir={output_dir}'
        print(f'{iteration}. Running command: {command}')
        try:
            run_bash_command(command)
            remove_from_error_file(error_file, qid, player_ids)
        except Exception as e:
            print(f'#### Error occured in competition {qid} {player_ids}: \n{str(e)}\n')


def runner_2of5(output_dir, positions_file='./data/2of5_competition/documents.positions'):
    error_file = output_dir + 'error_file.txt'
    qid_list = sorted(get_queries(positions_file))
    iteration = 0
    for dummy_bot in [1, 2]:
        for qid in qid_list:
            iteration += 1

            if exists(output_dir + 'trec_files/trec_file_{}_{}'.format(qid, dummy_bot)):
                print('Competition qid={} dummy_bot={} has already been ran'.format(qid, dummy_bot))
                continue

            command = f'python main.py --mode=2of5 --qid={qid} --dummy_bot={dummy_bot}' \
                      f' --output_dir={output_dir}'
            print(f'{iteration}. Running command: {command}')
            try:
                run_bash_command(command)
            except Exception as e:
                print(f'#### Error occured in competition {qid} {dummy_bot}:\n{str(e)}\n')
                log_error(error_file, qid, dummy_bot=dummy_bot)


def main():
    mode = sys.argv[1]
    output_dir = './output/{}/'.format(sys.argv[2])

    if mode == '2of2':
        runnner_2of2(output_dir)
    elif mode == 'rerun_2of2':
        rerun_errors_2of2(output_dir)
    elif mode == '2of5':
        runner_2of5(output_dir)
    else:
        print(f'Illegal mode {mode}')


if __name__ == '__main__':
    main()
