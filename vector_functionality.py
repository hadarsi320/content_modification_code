from krovetzstemmer import Stemmer
import numpy as np
import math
from utils import get_java_object, clean_texts


def clean_sentence(sentence):
    # sw = set(nltk.corpus.stopwords.words('english'))
    sw = []
    return [token for token in sentence.rstrip().split() if token not in sw]


def get_term_frequency(text, term):
    stemmer = Stemmer()
    return [stemmer.stem(token) for token in text.split()].count(term)


def query_term_freq(mode, text, query):
    if len(text.replace(' ', '')) == 0:
        # print("EMPTY TEXT")
        return 0
    if len(query.split("_")) > 1:
        freqs = [get_term_frequency(text, q) / len(text.split()) for q in query.split("_")]
    else:
        freqs = [get_term_frequency(text, q) / len(text.split()) for q in query.split()]
    if mode == "max":
        return max(freqs)
    if mode == "min":
        return min(freqs)
    if mode == "avg":
        return np.mean(freqs)
    if mode == "sum":
        return sum(freqs)


def dict_norm(dict):
    sum = 0
    for token in dict:
        sum += (float(dict[token]) ** 2)
    return sum


def dict_cosine_similarity(d1, d2):
    """
    cosine similarity between document vectors represented as dictionaries
    """
    sumxx = dict_norm(d1)
    sumyy = dict_norm(d2)
    if sumxx == 0 or sumyy == 0:
        return 0
    sumxy = 0
    shared_token = set(d1.keys()).intersection(set(d2.keys()))
    for token in shared_token:
        tfidf1 = float(d1[token])
        tfidf2 = float(d2[token])
        sumxy += tfidf1 * tfidf2
    return sumxy / math.sqrt(sumyy * sumxx)


def cosine_similarity(v1, v2):
    if v1 is None or v2 is None:
        return 0
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    if sumxx == 0 or sumyy == 0:
        return 0
    return sumxy / math.sqrt(sumxx * sumyy)


def get_text_centroid(text, model, stemmer=False):
    sum_vector = None
    denom = 0
    if stemmer:
        stem = Stemmer()
    for token in clean_sentence(text):
        if stemmer:
            token = stem.stem(token)
        try:
            vector = model.wv[token]
        except KeyError:
            continue
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector += vector
        denom += 1
    if sum_vector is None:
        return None
    return sum_vector / denom


def calculate_similarity_to_docs_centroid_tf_idf(text_tfidf_fname, top_docs_tfidf):
    summary_tfidf = get_java_object(text_tfidf_fname)
    return dict_cosine_similarity(summary_tfidf, top_docs_tfidf)


def embedding_similarity(s1, s2, model, stemmer=None):
    centroid1 = get_text_centroid(s1, model, stemmer)
    centroid2 = get_text_centroid(s2, model, stemmer)
    if centroid1 is None or centroid2 is None:
        return 0
    return cosine_similarity(centroid1, centroid2)


def document_tfidf_similarity(f1, f2):
    f1_dict = get_java_object(f1)
    f2_dict = get_java_object(f2)
    return dict_cosine_similarity(f1_dict, f2_dict)


def get_semantic_docs_centroid(doc_texts, doc_names, model, stemmer=None):
    sum_vector = None
    for doc in doc_names:
        text = doc_texts[doc]
        vector = get_text_centroid(clean_texts(text), model, stemmer)
        if sum_vector is None:
            sum_vector = np.zeros(vector.shape[0])
        sum_vector = sum_vector + vector
    if sum_vector is None:
        return None
    return sum_vector / len(doc_names)


def calculate_semantic_similarity_to_top_docs(text, top_docs, doc_texts, model, stemmer=None):
    summary_vector = get_text_centroid(clean_texts(text), model, stemmer)
    top_docs_centroid_vector = get_semantic_docs_centroid(doc_texts, top_docs, model, stemmer)
    return cosine_similarity(summary_vector, top_docs_centroid_vector)


def normalize_dict(dict, n):
    for token in dict:
        dict[token] = float(dict[token]) / n
    return dict


def document_centroid(document_vectors):
    centroid = {}
    for doc in document_vectors:
        centroid = add_dict(centroid, doc)
    return normalize_dict(centroid, len(document_vectors))


def add_dict(d1, d2):
    for token in d2:
        if token in d1:
            d1[token] = float(d1[token]) + float(d2[token])
        else:
            d1[token] = float(d2[token])
    return d1
