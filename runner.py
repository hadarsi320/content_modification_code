from utils import reverese_query, ensure_dir
from collections import defaultdict
from itertools import combinations
from os.path import exists
from gen_utils import run_and_print, run_bash_command
from random import sample


def get_competitors_dict(trec_file):
    competitors_dict = defaultdict(list)
    with open(trec_file) as f:
        for line in f:
            qrid = line.split()[0]
            epoch, qid = reverese_query(qrid)
            if epoch != '01':
                continue
            competitor = line.split()[2].split('-')[-1]
            competitors_dict[qid].append(competitor)
    return dict(competitors_dict)


def log_error(error_file, qid, comptitors):
    ensure_dir(error_file)
    with open(error_file, 'a') as f:
        f.write(f'Error in {qid} with competitors {", ".join(comptitors)}\n')


if __name__ == '__main__':
    trec = './trecs/trec_file_original_sorted.txt'
    output_dir = './output/run_29_9/'
    error_file = output_dir + 'error_file.txt'

    competitors = get_competitors_dict(trec)
    competitors_combinations = {qid: list(combinations(competitors[qid], 2)) for qid in competitors}
    iteration = 0
    for i in range(10):
        for qid in competitors_combinations:
            iteration += 1
            curr_comp = sorted(competitors_combinations[qid][i])
            if exists(output_dir + 'similarity_results/similarity_{}_{}.txt'.format(qid, ','.join(curr_comp))):
                print('Competition qid={} competitors={} has already been ran'.format(qid, ', '.join(curr_comp)))
                continue
            command = f'python main.py --mode=2of2 --qid={qid} --competitors={",".join(curr_comp)}' \
                      f' --output_dir={output_dir}'
            print('{}. Running command: {}'.format(iteration, command))
            try:
                run_bash_command(command)
            except Exception as e:
                print(f'#### Error occured in competition {qid} {", ".join(curr_comp)}: \n{str(e)}\n')
                log_error(error_file, qid, curr_comp)

    # for qid in competitors_combinations:
    #     competitors = sample(competitors_combinations[qid], 1)[0]
    #     command = 'python 2of2_competition.py --qid={} --competitors={} --output_dir=./output/run_26_9/ -r 8'\
    #         .format(qid, ','.join(competitors))
    #     run_and_print(command)
    #     break
