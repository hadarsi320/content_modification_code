from utils import *


def generate_dataset(local_dir, use_raifer_data):
    document_workingset_file = local_dir + 'doc_ws.txt'
    doc_tfidf_dir = local_dir + 'doc_tf_idf/'
    index = local_dir + 'index'

    if use_raifer_data:
        trectext_file = raifer_trectext_file
        trec_reader = TrecReader(trec_file=raifer_trec_file)
    else:
        trectext_file = goren_trectext_file
        trec_reader = TrecReader(positions_file=goren_positions_file)

    trec_texts = utils.read_trectext_file(trectext_file)
    stopwords = open(stopwords_file).read().split('\n')[:-1]


def main():
    pass


if __name__ == '__main__':
    main()
