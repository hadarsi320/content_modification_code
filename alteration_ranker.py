import utils


def main():
    trec_file = utils.read_trec_file('data/trec_file_original_sorted.txt')
    trectext_file = utils.read_trectext_file('data/documents.trectext')

    for epoch in trec_file:
        for qid in trec_file[epoch]:
            pass

    print('')


if __name__ == '__main__':
    main()
