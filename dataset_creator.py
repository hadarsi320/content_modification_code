import sys
import os
from utils import ensure_dirs


def read_labels(fname):
    stats = {}
    with open(fname) as file:
        for line in file:
            query = line.split()[0]
            doc = line.split()[2]
            label = float(line.split()[3].rstrip())
            if query not in stats:
                stats[query] = {}
            stats[query][doc] = label
        return stats


def normalize_lables(labels, weight):
    for qid in labels:
        for doc in labels[qid]:
            labels[qid][doc] = labels[qid][doc] * weight
    return labels


def get_level(score):
    assert score >= 0
    if score < 2:
        demotion_level = 2
    elif score < 4:
        demotion_level = 1
    else:
        demotion_level = 0
    return demotion_level


def modify_seo_score_by_demotion(seo_scores, coherency_scores):
    new_scores = {}
    for qid in seo_scores:
        new_scores[qid] = {}
        for doc in seo_scores[qid]:
            current_score = seo_scores[qid][doc]
            coherency_score = coherency_scores[qid][doc]
            demotion_level = get_level(coherency_score)
            new_score = max(current_score - demotion_level, 0.0)
            new_scores[qid][doc] = new_score
    return new_scores


def create_harmonic_mean_score(seo_scores, coherency_scores, beta):
    new_scores = {}
    for qid in seo_scores:
        new_scores[qid] = {}
        for doc in seo_scores[qid]:
            epsilon = 0.0001
            current_score = seo_scores[qid][doc]
            coherency_score = coherency_scores[qid][doc]
            new_coherency_score = coherency_score * (4.0 / 5)
            numerator = (1 + beta ** 2) * new_coherency_score * current_score
            denominator = (beta ** 2) * new_coherency_score + current_score
            denominator += epsilon
            harmonic_mean = float(numerator) / denominator
            new_scores[qid][doc] = harmonic_mean
    return new_scores


def create_weighted_mean_score(seo_scores, coherency_scores, beta):
    new_scores = {}
    for qid in seo_scores:
        new_scores[qid] = {}
        for doc in seo_scores[qid]:
            current_score = seo_scores[qid][doc]
            coherency_score = coherency_scores[qid][doc]
            new_coherency_score = coherency_score * (4.0 / 5)
            new_score = current_score * beta + new_coherency_score * (1 - beta)
            new_scores[qid][doc] = new_score
    return new_scores


def rewrite_features(labels, features_fname, output_features_fname):
    ensure_dirs(output_dir)
    with open(output_features_fname, 'w') as output:
        with open(features_fname) as base_features:
            for data_point in base_features:
                name = data_point.split(" # ")[1].rstrip()
                query = data_point.split()[1].replace("qid:", "")
                label = str(labels[query][name])
                new_data_point = label + " " + " ".join(data_point.split()[1:]).rstrip() + "\n"
                output.write(new_data_point)


def rewrite_qrels(lables, output_qrels_fname):
    ensure_dirs(output_qrels_fname)
    with open(output_qrels_fname, 'w') as output:
        for qid in lables:
            for doc in lables[qid]:
                label = str(lables[qid][doc])
                output.write(qid + " 0 " + doc + " " + label + "\n")


if __name__ == "__main__":
    output_dir = sys.argv[1]
    label_strategy = sys.argv[2]
    seo_qrels = sys.argv[3]
    coherency_qrels = sys.argv[4]
    feature_fname = sys.argv[5]

    qrels_base_dir = output_dir + 'qrels_sets/'
    features_base_dir = output_dir + 'feature_sets/'

    seo_scores = read_labels(seo_qrels)
    coherency_scores = read_labels(coherency_qrels)

    if label_strategy == "demotion":
        combined_labels = modify_seo_score_by_demotion(seo_scores, coherency_scores)
        rewrite_features(combined_labels, feature_fname,
                         output_features_fname=features_base_dir+"demotion/demotion_features")
        rewrite_qrels(combined_labels,
                      output_qrels_fname=qrels_base_dir+"demotion/demotion_qrels")
    elif label_strategy == "harmonic":
        betas = [0, 0.5, 1, 2, 1000, 100000, 1000000000]
        for beta in betas:
            combined_labels = create_harmonic_mean_score(seo_scores, coherency_scores, beta)
            rewrite_features(combined_labels, feature_fname,
                             output_features_fname=features_base_dir+"harmonic/harmonic_features_"+str(beta))
            rewrite_qrels(combined_labels,
                          output_qrels_fname=qrels_base_dir+"harmonic/harmonic_qrels_"+str(beta))
    elif label_strategy == "weighted":
        betas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for beta in betas:
            combined_labels = create_weighted_mean_score(seo_scores, coherency_scores, beta)
            rewrite_features(combined_labels, feature_fname,
                             output_features_fname=features_base_dir+"weighted/weighted_features_"+str(beta))
            rewrite_qrels(combined_labels,
                          output_qrels_fname=qrels_base_dir+"weighted/weighted_qrels_"+str(beta))
    else:
        print('invalid label strategy: {}'.format(label_strategy))
