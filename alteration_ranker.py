import os

import bot_competition
import readers
import utils


def main():
    trec_file = 'data/trec_file_original_sorted.txt'
    trectext_file = 'data/documents.trectext'
    swig_path = '/lv_local/home/hadarsi/indri-5.6/swig/obj/java/'
    base_index = '/lv_local/home/hadarsi/work_files/clueweb_index/'
    indri_path = '/lv_local/home/hadarsi/indri/'

    local_dir = 'tmp'
    index = local_dir + 'index'
    document_workingset_file = local_dir + 'doc_ws'
    doc_tfidf_dir = local_dir + 'doc_tf_idf'

    changed = []
    x = []
    y = []

    trec_reader = readers.TrecReader(trec_file)
    trec_texts = utils.read_trectext_file(trectext_file)

    for epoch in trec_reader.get_epochs():
        next_epoch = utils.get_next_epoch(epoch)
        if next_epoch not in trec_reader.get_epochs():
            break

        for qid in trec_reader.get_queries():
            doc_id = trec_reader[epoch][qid][0]
            next_doc_id = utils.get_next_doc_id(doc_id)

            # create x
            os.makedirs(local_dir)
            utils.create_index('data/documents.trectext', new_index_name=index, indri_path=indri_path)
            utils.create_documents_workingset(document_workingset_file, competitors, qid, epoch=1)
            bot_competition.generate_document_tfidf_files(document_workingset_file, output_dir=doc_tfidf_dir,
                                                          swig_path=swig_path, base_index=base_index,
                                                          new_index=index)

            # Create y
            next_rank = trec_reader[next_epoch][qid].index(next_doc_id)
            y.append(next_rank == 0)

            # doc_text = trec_texts[doc_id]
            # next_doc_text = trec_texts[utils.get_next_doc_id(doc_id)]
            # changed.append(doc_text == next_doc_text)

    # print(f'{sum(changed)} top documents were changed out of {len(changed)} total top documents,'
    #       f' on average {sum(changed) / len(changed):.3f}')
    # print(f'{sum(y)} top documents stayed first out of {len(y)} total top documents,'
    #       f' on average {sum(y) / len(y):.3f}')


if __name__ == '__main__':
    main()
