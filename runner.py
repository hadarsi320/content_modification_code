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
    output_dir = './output/run_26_9/'
    error_file = output_dir + 'error_file.txt'

    competitors = get_competitors_dict(trec)
    competitors_combinations = {qid: list(combinations(competitors[qid], 2)) for qid in competitors}
    for i in range(10):
        for qid in competitors_combinations:
            curr_comp = competitors_combinations[qid][i]
            if exists(output_dir + 'trec_files/trec_file{}_{}'.format(qid, ','.join(curr_comp))):
                continue
            command = f'python main.py --qid={qid} --competitors={",".join(curr_comp)} --output_dir={output_dir}'
            try:
                run_and_print(command)
            except Exception:
                print(f'#### Error occured in competition {qid} {", ".join(curr_comp)}')
                log_error(error_file, qid, curr_comp)

    # for qid in competitors_combinations:
    #     competitors = sample(competitors_combinations[qid], 1)[0]
    #     command = 'python main.py --qid={} --competitors={} --output_dir=./output/run_26_9/ -r 8'\
    #         .format(qid, ','.join(competitors))
    #     run_and_print(command)
    #     break
