from utils import reverese_query
from collections import defaultdict
from itertools import combinations
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


if __name__ == '__main__':
    trec = './trecs/trec_file_original_sorted.txt'
    competitors = get_competitors_dict(trec)
    competitors_combinations = {qid: list(combinations(competitors[qid], 2)) for qid in competitors}
    # for i in range(10):
    #     for qid in competitors_combinations:
    #         command = 'python main.py --qid=' + qid + ' --competitors=' + ','.join(competitors_combinations[qid][i])
    #         run_and_print(command)
    for qid in competitors_combinations:
        competitors = sample(competitors_combinations[qid], 1)[0]
        command = 'python main.py --qid={0} --competitors={1} --output_dir=./output/run_25.9/'\
            .format(qid, ','.join(competitors))
        run_and_print(command)
        break

