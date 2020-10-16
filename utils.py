import logging
import os
import re
import shutil
import sys
from collections import defaultdict

import gensim
import javaobj
from deprecated import deprecated

from gen_utils import run_bash_command, run_command, run_and_print
import xml.etree.ElementTree as ET
from lxml import etree
from nltk import sent_tokenize


def create_features_file_diff(features_dir, base_index_path, new_index_path, new_features_file,
                              working_set_file, scripts_path, swig_path, stopwords_file, queries_text_file):
    """
    Creates a feature file via a given index and a given working set file
    """
    logger = logging.getLogger(sys.argv[0])
    run_bash_command("rm -r " + features_dir)  # 'Why delete this directory and then check if it exists?'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    ensure_dirs(new_features_file)

    command = f'java -Djava.library.path={swig_path} -cp seo_indri_utils.jar LTRFeatures {base_index_path} ' \
              f'{new_index_path} {stopwords_file} {queries_text_file} {working_set_file} {features_dir}'
    run_and_print(command, command_name='LTRFeatures')

    command = f"perl {scripts_path}generate.pl {features_dir} {working_set_file}"
    logger.info(command)
    run_bash_command(command)

    command = f"mv features {new_features_file}"
    logger.info(command)
    run_bash_command(command)

    run_bash_command("mv featureID " + os.path.dirname(new_features_file))
    return new_features_file


def read_trec_file(trec_file, current_round=None, current_qid=None, competitor_list=None):
    stats = defaultdict(dict)
    with open(trec_file) as file:
        for line in file:
            doc_id = line.split()[2]
            epoch, qid, pid = parse_doc_id(doc_id)
            if (current_round and int(epoch) > int(current_round)) or \
                    (current_qid and current_qid != qid) or \
                    (competitor_list and pid not in competitor_list):
                continue
            if qid not in stats[epoch]:
                stats[epoch][qid] = []
            stats[epoch][qid].append(doc_id)
    return dict(stats)


def read_raw_trec_file(trec_file):
    stats = defaultdict(list)
    with open(trec_file) as file:
        for line in file:
            last_qrid = line.split()[0]
            doc_id = line.split()[2]
            stats[last_qrid].append(doc_id)
    return dict(stats)


def read_competition_trec_file(trec_file):
    stats = defaultdict(list)
    with open(trec_file) as file:
        for line in file:
            doc_id = line.split()[2]
            epoch, _, pid = parse_doc_id(doc_id)
            stats[epoch].append(pid)
    return dict(stats)


def read_positions_file(positions_file):
    qid_list = get_query_ids(positions_file)
    stats = {qid: {epoch: [None]*5 for epoch in range(1, 5)} for qid in qid_list}
    with open(positions_file, 'r') as f:
        for line in f:
            doc_id = line.split()[2]
            epoch, qid, pid = parse_doc_id(doc_id)
            epoch = int(epoch)
            position = int(line.split()[3]) - 1
            stats[qid][epoch][position] = pid
    return stats


def create_trectext_file(document_texts, trectext_fname, working_set_fname=None):
    """
    creates trectext document from a given text file
    """
    trectext_dir = os.path.dirname(trectext_fname)
    if not os.path.exists(trectext_dir):
        os.makedirs(trectext_dir)

    with open(trectext_fname, "w", encoding="utf-8") as f:
        f.write('<DATA>\n')
        query_to_docs = defaultdict(list)
        for document in sorted(document_texts):
            text = document_texts[document]
            query = document.split("-")[2]
            query_to_docs[query].append(document)

            f.write('<DOC>\n')
            f.write('<DOCNO>' + document + '</DOCNO>\n')
            f.write('<TEXT>\n')
            f.write(text.rstrip().strip('\n'))
            f.write('\n</TEXT>\n')
            f.write('</DOC>\n')
        f.write('</DATA>\n')

    # what is the purpose of creating this file?
    if working_set_fname:
        with open(working_set_fname, 'w') as f:
            for query, docnos in query_to_docs.items():
                i = 1
                for docid in docnos:
                    f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
                    i += 1
    return trectext_fname


def update_trectext_file(trectext_file, old_documents, new_documents):
    logger = logging.getLogger(sys.argv[0])
    create_trectext_file({**old_documents, **new_documents}, trectext_file)
    logger.info('Trectext file updated')


def create_index(trectext_file, new_index_name, indri_path):
    """
    Parse the trectext file given, and create an index.
    """
    if os.path.exists(new_index_name):
        shutil.rmtree(new_index_name)

    corpus_class = 'trectext'
    memory = '1G'
    stemmer = 'krovetz'
    ensure_dirs(new_index_name)
    command = f'{indri_path}bin/IndriBuildIndex -corpus.path={trectext_file} -corpus.class={corpus_class} ' \
              f'-index={new_index_name} -memory={memory} -stemmer.name={stemmer}'
    run_and_print(command, command_name='IndriBuildIndex')
    return new_index_name


def merge_indices(merged_index, new_index_name, base_index, home_path, indri_path):
    """
    merges two different indri indices into one
    """
    # new_index_name = home_path +'/' + index_path +'/' + new_index_name
    if os.path.exists(merged_index):
        os.remove(merged_index)
    ensure_dirs(merged_index)
    command = home_path + "/" + indri_path + '/bin/dumpindex ' + merged_index + ' merge ' + new_index_name + ' ' + \
              base_index
    print("##merging command:", command + "##", flush=True)
    out = run_bash_command(command)
    print("merging command output:" + str(out), flush=True)
    return new_index_name


def create_trec_eval_file(results, trec_file):
    ensure_dirs(trec_file)
    with open(trec_file, 'w') as f:
        for query in results:
            for doc in results[query]:
                f.write(query + " Q0 " + doc + " " + str(0) + " " + str(results[query][doc]) + " seo_task\n")
    return trec_file


def order_trec_file(trec_file):
    """
    Sorts a trec file
    :param trec_file: trec file path
    :return: the path to the sorted trec file
    """
    final = trec_file.replace(".txt", "") + '_sorted_txt'
    command = "sort -k1,1n -k5nr -k2,1 " + trec_file + " > " + final
    for line in run_command(command):
        print(line)
    return final


def retrieve_scores(test_indices, queries, score_file):
    results = defaultdict(dict)
    with open(score_file) as scores:
        for i, score in enumerate(scores):
            query = queries[i]
            doc = test_indices[i]
            results[query][doc] = float(score.split()[2].rstrip())
        return dict(results)


def create_index_to_doc_name_dict(data_set_file):
    doc_name_index = {}
    index = 0
    with open(data_set_file) as ds:
        for line in ds:
            rec = line.split("# ")
            doc_name = rec[1].rstrip()
            doc_name_index[index] = doc_name
            index += 1
        return doc_name_index


def create_index_to_query_dict(data_set_file):
    query_index = {}
    index = 0
    with open(data_set_file) as ds:
        for line in ds:
            rec = line.split()
            query = rec[1].split(":")[1]
            query_index[index] = query
            index += 1
        return query_index


def run_model(features_file, jar_path, score_file, model_path):
    ensure_dirs(score_file)
    run_bash_command('touch ' + score_file)
    command = "java -jar " + jar_path + " -load " + model_path + " -rank " + features_file + " -score " + \
              score_file
    run_and_print(command)
    return score_file


def get_past_winners(ranked_lists, epoch, query):
    past_winners = []
    for iteration in range(int(epoch)):
        current_epoch = str(iteration + 1).zfill(2)
        past_winners.append(ranked_lists[current_epoch][query][0])
    return past_winners


def parse_qrid(qrid):
    """
    :return: epoch, qrid
    """
    epoch = str(qrid)[-2:]
    qrid = str(qrid)[:-2].zfill(3)
    return epoch, qrid


def fix_format(doc_id):
    epoch, qid, pid = parse_doc_id(doc_id)
    return get_doc_id(epoch, qid, pid)


def load_trectext_file(filename, qid=None):
    parser = etree.XMLParser(recover=True)
    tree = ET.parse(filename, parser=parser)
    root = tree.getroot()
    docs = {}
    for doc in root:
        epoch = last_qid = pid = None
        for att in doc:
            if att.tag == "DOCNO":
                doc_id = fix_format(att.text)
                epoch, last_qid, pid = parse_doc_id(doc_id)
                if qid and last_qid != qid:
                    break
            else:
                docs[get_doc_id(epoch, last_qid, pid.replace('_', ''))] = att.text
    return docs


def get_java_object(obj_file):
    with open(obj_file, 'rb') as fd:
        obj = javaobj.load(fd)
    return obj


def clean_texts(text):
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = text.replace("?", " ")
    text = text.replace("]", "")
    text = text.replace("[", "")
    text = text.replace("}", "")
    text = text.replace("{", "")
    text = text.replace("+", " ")
    text = text.replace("~", " ")
    text = text.replace("^", " ")
    text = text.replace("#", " ")
    text = text.replace("$", " ")
    text = text.replace("!", "")
    text = text.replace("|", " ")
    text = text.replace("%", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("\\", " ")
    text = text.replace("*", " ")
    text = text.replace("&", " ")
    text = text.replace(";", " ")
    text = text.replace("`", "")
    text = text.replace("'", "")
    text = text.replace("’", "")
    text = text.replace("@", " ")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    text = text.replace("/", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    # my additions
    text = text.replace("\t", " ") \
        .strip()
    return text.lower()


def transform_query_text(queries_raw_text):
    """
    Transforms all queries from '#combine( [query text] )' to '[query text]'
    :param queries_raw_text: query dictionary
    :return: transformed query dictionary
    """
    transformed = {}
    for qid in queries_raw_text:
        transformed[qid] = queries_raw_text[qid].replace("#combine( ", "").replace(" )", "")
    return transformed


def read_queries_file(queries_file, current_qrid=None):
    # consider receiving just the qid, since the round doesn't matter
    """
    reads a queries xml file
    :param queries_file: location of the queries file
    :return: dictionary of the sort {query id: query text}
    """
    last_qrid = None
    stats = {}
    with open(queries_file) as file:
        for line in file:
            if "<number>" in line:
                last_qrid = line.replace('<number>', '').replace('</number>', "").split("_")[
                    0].rstrip().replace("\t", "").replace(" ", "")

            if '<text>' in line and (not last_qrid or last_qrid == current_qrid):
                stats[last_qrid] = line.replace('<text>', '').replace('</text>', '').rstrip().replace("\t", "")
    return stats


def get_query_text(queries_file, current_qid):
    with open(queries_file) as file:
        for line in file:
            if "<number>" in line:
                qrid = line.replace('<number>', '').replace('</number>', "").split("_")[0].rstrip() \
                    .replace("\t", "").replace(" ", "")
                _, qid = parse_qrid(qrid)
            if '<text>' in line and qid == current_qid:
                query_text = line.replace('<text>', '').replace('</text>', '').rstrip().replace("\t", "") \
                    .replace("#combine( ", "").replace(" )", "")
                return query_text
    raise Exception('No query with qid={} in file {}'.format(current_qid, queries_file))


def get_learning_data_path(learning_data_dir, label_aggregation_method, label_aggregation_b):
    learning_data_path = learning_data_dir + label_aggregation_method + '/' + label_aggregation_method + '_features'
    if not label_aggregation_method == 'demotion':
        learning_data_path += f'_{label_aggregation_b}'
    return learning_data_path


def get_model_name(label_aggregation_method: str, label_aggregation_b: str, svm_rank_c: str):
    return 'svm_rank_model_' + label_aggregation_method + '_' + \
           (f'b={label_aggregation_b}_' if label_aggregation_method != 'demotion' else '') \
           + 'c=' + svm_rank_c + '.dat'


@deprecated(reason='This was created a while ago')
def print_and_delete(file_path):
    with open(file_path) as f:
        print(f.read())
    os.remove(file_path)


def get_qrid(qid: str, epoch: int):
    return qid.lstrip('0') + str(epoch).zfill(2)


def parse_doc_id(doc_id: str):
    """
    :param doc_id: an id of the form ROUND-[epoch]-[qid]-[pid]
    :return: (epoch, qid, pid)
    """
    epoch, qid, pid = doc_id.strip(' ').split('-')[1:]
    epoch = epoch.zfill(2)
    return epoch, qid, pid


def get_doc_id(epoch: int, qid, player_id):
    return f'ROUND-{int(epoch):02d}-{qid}-{player_id}'


def generate_pair_name(pair):
    out_ = str(int(pair.split("_")[1]))
    in_ = str(int(pair.split("_")[2]))
    return pair.split("$")[1].split("_")[0] + "_" + out_ + "_" + in_


def create_documents_workingset(output_file, epoch, qid, competitor_list):
    ensure_dirs(output_file)
    with open(output_file, 'w') as f:
        for competitor in competitor_list:
            line = get_qrid(qid, epoch) + ' Q0 ' + get_doc_id(epoch, qid, competitor) + ' 0 0 indri\n'
            f.write(line)


def ensure_dirs(*args):
    for file in args:
        dir_name = os.path.dirname(file)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('{} Creating directory: {}'.format('#' * 20, dir_name))


def tokenize_document(document):
    return [clean_texts(sentence) for sentence in sent_tokenize(document)]


def is_file_empty(file):
    with open(file) as f:
        text = f.read()
    return text == ''


def complete_sim_file(similarity_file, total_rounds):
    lines = 0
    with open(similarity_file, 'r') as f:
        for line in f:
            if len(line) > 0:
                lines += 1
    with open(similarity_file, 'a') as f:
        for i in range(total_rounds - lines + 1):
            # replace this with actual similarity
            f.write('{}\t{}\t{}\n'.format(lines + i, 1, 1))


def xor(a, b):
    return bool(a) != bool(b)


def normalize_dict_len(dictionary):
    max_len = max(len(dictionary[key]) for key in dictionary)
    irregular_keys = [key for key in dictionary if len(dictionary[key]) < max_len]
    for key in irregular_keys:
        dictionary.pop(key)


def get_next_doc_id(doc_id):
    epoch, qid, pid = parse_doc_id(doc_id)
    epoch = int(epoch)+1
    return get_doc_id(epoch, qid, pid)


def get_next_qrid(qrid):
    epoch, qid = parse_qrid(qrid)
    return get_qrid(qid, int(epoch)+1)


def get_query_ids(file):
    """
    :param file: a trec or positions file
    :return: all qids in file
    """
    qid_list = []
    with open(file) as f:
        for line in f:
            qid = line.split()[2].split('-')[2]
            if qid not in qid_list:
                qid_list.append(qid)
    return sorted(qid_list)


def load_word_embedding_model(model_file):
    return gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True, limit=700000)