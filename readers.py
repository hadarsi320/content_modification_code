import random
from collections import defaultdict

import os

import utils


class TrecReader:
    def __init__(self, **kwargs):
        self.__queries = set()
        self.__epochs = set()
        self.__qrid_list = set()

        if 'trec_file' in kwargs:
            self.__ranked_list = self.__read_trec_file(kwargs.pop('trec_file'))
        elif 'trec_dir' in kwargs:
            self.__ranked_list = self.__read_trec_dir(kwargs.pop('trec_dir'))
        else:
            raise ValueError('No value given to TrecReader')

    def __read_trec_file(self, trec_file):
        ranked_list = defaultdict(dict)
        with open(trec_file) as file:
            for line in file:
                doc_id = line.split()[2]
                epoch, qid, _ = utils.parse_doc_id(doc_id)

                self.__epochs.add(epoch)
                self.__queries.add(qid)

                if qid not in ranked_list[epoch]:
                    ranked_list[epoch][qid] = []
                ranked_list[epoch][qid].append(doc_id)
        return dict(ranked_list)

    def __read_raw_trec_file(self, trec_file):
        ranked_list = defaultdict(list)
        with open(trec_file) as file:
            for line in file:
                qrid = line.split()[0]
                doc_id = line.split()[2]

                self.__qrid_list.add(qrid)
                ranked_list[qrid].append(doc_id)
        return dict(ranked_list)

    def __read_trec_dir(self, trec_dir):
        ranked_list = defaultdict(dict)
        trec_files = sorted(os.listdir(trec_dir))
        for trec_fname in trec_files:
            trec_file = f'{trec_dir}/{trec_fname}'
            qid = '_'.join(trec_fname.split('_')[-2:])
            self.__queries.add(qid)
            with open(trec_file, 'r') as f:
                for line in f:
                    doc_id = line.split()[2]
                    epoch, _, pid = utils.parse_doc_id(doc_id)
                    self.__epochs.add(epoch)
                    if qid not in ranked_list[epoch]:
                        ranked_list[epoch][qid] = []
                    ranked_list[epoch][qid].append(doc_id)
        return ranked_list

    def __getitem__(self, epoch):
        epoch = str(epoch).zfill(2)
        return self.__ranked_list[epoch]

    def add_epoch(self, ranked_lists: dict):
        last_epoch = max(self.__epochs)
        new_epoch = utils.get_next_epoch(last_epoch)
        self.__ranked_list[new_epoch] = ranked_lists
        self.__epochs.add(new_epoch)

    def __iter__(self):
        return iter(sorted(self.__epochs))

    def epochs(self):
        return sorted(self.__epochs)

    def queries(self):
        return sorted(self.__queries)

    def num_epochs(self):
        return len(self.__epochs)

    def num_queries(self):
        return len(self.__queries)

    def get_pids(self, qid):
        epoch = min(self.__epochs)
        player_ids = [utils.parse_doc_id(doc_id)[2] for doc_id in self.__ranked_list[epoch][qid]]
        return player_ids

    def max_rank(self):
        return max(len(self.get_pids(qid)) for qid in self.__queries)


if __name__ == '__main__':
    # trec = TrecReader(trec_file='data/trec_file_original_sorted.txt')
    trec_reader = TrecReader(trec_dir='results/1of5_12_21_highest_rated_inferiors_alteration_classifier/trec_files')
    print(random.choice(trec_reader.queries()))
