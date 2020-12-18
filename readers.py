from collections import defaultdict

from utils import parse_doc_id


class TrecReader:
    def __init__(self, trec_file, raw=False):
        self.queries = set()
        self.epochs = set()
        self.qrid_list = set()

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

                self.epochs.add(epoch)
                self.queries.add(qid)

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

                self.qrid_list.add(qrid)
                self.__ranked_list[qrid].append(doc_id)
        self.__ranked_list = dict(self.__ranked_list)

    def __getitem__(self, item):
        return self.__ranked_list[item]

    # def __iter__(self):
    #     return iter()


if __name__ == '__main__':
    trec = TrecReader('data/trec_file_original_sorted.txt')
    for i in trec:
        print(i)
