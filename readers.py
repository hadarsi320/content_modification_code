from collections import defaultdict

import utils
from utils import parse_doc_id


class TrecReader:
    def __init__(self, trec_file, raw=False):
        self.__queries = set()
        self.__epochs = set()
        self.__qrid_list = set()

        if raw:
            self.__read_raw_trec_file(trec_file)
        else:
            self.__read_trec_file(trec_file)

    def __read_trec_file(self, trec_file):
        self.__ranked_list = defaultdict(dict)
        with open(trec_file) as file:
            for line in file:
                doc_id = line.split()[2]
                epoch, qid, _ = parse_doc_id(doc_id)

                self.__epochs.add(epoch)
                self.__queries.add(qid)

                if qid not in self.__ranked_list[epoch]:
                    self.__ranked_list[epoch][qid] = []
                self.__ranked_list[epoch][qid].append(doc_id)
        self.__ranked_list = dict(self.__ranked_list)

    def __read_raw_trec_file(self, trec_file):
        self.__ranked_list = defaultdict(list)
        with open(trec_file) as file:
            for line in file:
                qrid = line.split()[0]
                doc_id = line.split()[2]

                self.__qrid_list.add(qrid)
                self.__ranked_list[qrid].append(doc_id)
        self.__ranked_list = dict(self.__ranked_list)

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

    def get_epochs(self):
        return sorted(self.__epochs)

    def get_queries(self):
        return sorted(self.__queries)

    def get_num_epochs(self):
        return len(self.__epochs)

    def get_num_queries(self):
        return len(self.__queries)

    def get_player_ids(self, qid):
        epoch = min(self.__epochs)
        player_ids = [parse_doc_id(doc_id)[2] for doc_id in self.__ranked_list[epoch][qid]]
        return player_ids


if __name__ == '__main__':
    trec = TrecReader('data/trec_file_original_sorted.txt')
    for i in trec:
        print(i)
