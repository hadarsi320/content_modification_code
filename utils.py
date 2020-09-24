import os
import re
from collections import defaultdict
import javaobj

from gen_utils import run_bash_command, run_command
import xml.etree.ElementTree as ET
from lxml import etree


def create_features_file_diff(features_dir, base_index_path, new_index_path, new_features_file, working_set_file,
                              scripts_path, swig_path, stopwords_file, queries_text_file):
    """
    Creates a feature file via a given index and a given working set file
    """
    run_bash_command("rm -r " + features_dir)  # 'Why delete this directory and then check if it exists?'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(os.path.dirname(new_features_file)):
        os.makedirs(os.path.dirname(new_features_file))
    command = "java -Djava.library.path=" + swig_path + \
              " -cp seo_indri_utils.jar LTRFeatures " + base_index_path + " " + new_index_path + " " \
              + stopwords_file + " " + queries_text_file + " " + working_set_file + " " + features_dir
    print(command)
    out = run_bash_command(command)
    print(out)
    command = "perl " + scripts_path + "generate.pl " + features_dir + " " + working_set_file
    print(command)
    out = run_bash_command(command)
    print(out)
    command = "mv features " + new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv featureID " + os.path.dirname(new_features_file))
    return new_features_file


def read_trec_file(trec_file, current_round=None, current_qid=None, competitor_list=None):
    stats = defaultdict(dict)
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            epoch, query, competitor = doc.split('-')[1:]
            if (current_round and int(epoch) > int(current_round)) or \
                    (current_qid and current_qid != query) or \
                    (competitor_list and competitor not in competitor_list):
                continue
            if query not in stats[epoch]:
                stats[epoch][query] = []
            stats[epoch][query].append(doc)
    return dict(stats)


def read_raw_trec_file(trec_file):
    stats = {}
    with open(trec_file) as file:
        for line in file:
            doc = line.split()[2]
            query = line.split()[0]
            if query not in stats:
                stats[query] = []
            stats[query].append(doc)
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
            f.write(text.rstrip())
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


def append_to_trectext_file(trectext_file, document_texts):
    old_texts = load_trectext_file(trectext_file)
    create_trectext_file({**old_texts, **document_texts}, trectext_file)
    # with open(trectext_file, 'r') as f:
    #     trectext_lines = f.readlines()[:-1]
    # with open(trectext_file, "w", encoding="utf-8") as f:
    #     for line in trectext_lines:
    #         f.write(line)
    #     for document in document_texts:
    #         text = document_texts[document]
    #
    #         f.write('<DOC>\n')
    #         f.write('<DOCNO>' + document + '</DOCNO>\n')
    #         f.write('<TEXT>\n')
    #         f.write(text.rstrip())
    #         f.write('\n</TEXT>\n')
    #         f.write('</DOC>\n')
    #     f.write('</DATA>\n')


def create_index(trec_text_file, index, indri_path):
    """
    Parse the trectext file given, and create an index.
    """
    corpus_path = trec_text_file
    corpus_class = 'trectext'
    memory = '1G'
    stemmer = 'krovetz'
    index_path = os.path.dirname(index)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    # if not os.path.exists(home_path + index_path): what is this
    #     os.makedirs(home_path + index_path)
    command = indri_path + 'bin/IndriBuildIndex -corpus.path=' + corpus_path + ' -corpus.class=' + corpus_class + \
              ' -index=' + index + ' -memory=' + memory + ' -stemmer.name=' + stemmer
    print("##Running IndriBuildIndex command: " + command + "##", flush=True)
    out = run_bash_command(command)
    print("IndriBuildIndex output:" + out, flush=True)
    return index


def merge_indices(merged_index, new_index_name, base_index, home_path, indri_path):
    """
    merges two different indri indices into one
    """
    # new_index_name = home_path +'/' + index_path +'/' + new_index_name
    if os.path.exists(merged_index):
        os.remove(merged_index)
    if not os.path.exists(os.path.dirname(merged_index)):
        os.makedirs(os.path.dirname(merged_index))
    command = home_path + "/" + indri_path + '/bin/dumpindex ' + merged_index + ' merge ' + new_index_name + ' ' +  \
              base_index
    print("##merging command:", command + "##", flush=True)
    out = run_bash_command(command)
    print("merging command output:" + str(out), flush=True)
    return new_index_name


def create_trec_eval_file(results, trec_file):
    if not os.path.exists(os.path.dirname(trec_file)):
        os.makedirs(os.path.dirname(trec_file))
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
    if not os.path.exists(os.path.dirname(score_file)):
        os.makedirs(os.path.dirname(score_file))
    run_bash_command('touch ' + score_file)
    command = "java -jar " + jar_path + " -load " + model_path + " -rank " + features_file + " -score " + \
              score_file
    print(command)
    out = run_bash_command(command)
    print(out)
    return score_file


def get_past_winners(ranked_lists, epoch, query):
    past_winners = []
    for iteration in range(int(epoch)):
        current_epoch = str(iteration + 1).zfill(2)
        past_winners.append(ranked_lists[current_epoch][query][0])
    return past_winners


def reverese_query(qrid):
    epoch = str(qrid)[-2:]
    query = str(qrid)[:-2].zfill(3)
    return epoch, query


def load_trectext_file(filename):
    parser = etree.XMLParser(recover=True)
    tree = ET.parse(filename, parser=parser)
    root = tree.getroot()
    docs = {}
    for doc in root:
        name = ""
        for att in doc:
            if att.tag == "DOCNO":
                name = att.text
            else:
                docs[name] = att.text
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
                qid = reverese_query(qrid)[1]
            if '<text>' in line and qid == current_qid:
                query_text = line.replace('<text>', '').replace('</text>', '').rstrip().replace("\t", "") \
                    .replace("#combine( ", "").replace(" )", "")
                return query_text
    raise Exception('No query with such qid: {}'.format(current_qid))


def get_learning_data_path(learning_data_dir, label_aggregation_method, label_aggregation_b):
    learning_data_path = learning_data_dir + label_aggregation_method + '/' + label_aggregation_method + '_features'
    if not label_aggregation_method == 'demotion':
        learning_data_path += f'_{label_aggregation_b}'
    return learning_data_path


def get_model_name(label_aggregation_method: str, label_aggregation_b: str, svm_rank_c: str):
    return 'svm_rank_model_' + label_aggregation_method + '_' + \
           (f'b={label_aggregation_b}_' if label_aggregation_method != 'demotion' else '') \
           + 'c=' + svm_rank_c + '.dat'


# TODO consider deleting this
def print_and_delete(file_path):
    with open(file_path) as f:
        print(f.read())
    os.remove(file_path)


def get_qrid(qid: str, epoch: int):
    return qid.lstrip('0') + str(epoch).zfill(2)


def parse_trec_id(trec_id: str):
    return trec_id.strip(' ').split('-')[1:]


def generate_trec_id(epoch, qid, player_id):
    return f'ROUND-{epoch:02d}-{qid}-{player_id}'


def generate_pair_name(pair):
    out_ = str(int(pair.split("_")[1]))
    in_ = str(int(pair.split("_")[2]))
    return pair.split("$")[1].split("_")[0] + "_" + out_ + "_" + in_


def create_sentence_workingset(output_file, epoch, qid, competitor_list):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, 'w') as f:
        for competitor in competitor_list:
            line = get_qrid(qid, epoch) + ' Q0 ' + generate_trec_id(epoch, qid, competitor) + ' 0 0 indri'
            f.write(line + '\n')
