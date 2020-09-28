import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    similarity_dir = '/lv_local/home/hadarsi/pycharm_projects/content_modification_code/output/run_28_9/similarity_results/'
    plots_dir = './plots/'
    similarity_files = sorted(os.listdir(similarity_dir))

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    sim_lists = [], []
    for file in similarity_files:
        sim_list = [], []
        with open(similarity_dir+file) as f:
            for num, line in enumerate(f):
                if num == 0:
                    continue
                for i in [0, 1]:
                    sim_list[i].append(float(line.split('\t')[i+1]))
        for i in [0, 1]:
            sim_lists[i].append(sim_list[i])

    rounds = max(len(list) for list in sim_lists[0])
    lex_sim_matrix = np.array([list for list in sim_lists[0] if len(list) == rounds])
    embed_sim_matrix = np.array([list for list in sim_lists[1] if len(list) == rounds])


    averaged_mat = np.average(sim_matrix, axis=0)
    plt.plot(range(0, rounds), averaged_mat, 'o-')
    plt.xticks(range(0, rounds+1))
    plt.title(f'Similarity Measure Averaged Across {len(sim_matrix)} Competitions')
    plt.xlabel('Round')
    plt.ylabel('Cosine Similarity')
    # plt.savefig(plots_dir + 'similarity_plots.png')
    plt.show()

    # histogram
    alpha = 0.5
    bins = 12
    plt.hist(sim_matrix[:, 0], bins=bins, label='First Round', alpha=alpha)
    for i in range(0, sim_matrix.shape[1], 4):
        if i == 0 or i == sim_matrix.shape[1] - 1:
            continue
        plt.hist(sim_matrix[:, i], bins=bins, alpha=alpha, label='Round {}'.format(i))
    plt.hist(sim_matrix[:, -1], bins=bins, label='Last Round', alpha=alpha)
    plt.legend()
    plt.title('Similarity Histogram')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Counts')
    # plt.savefig(plots_dir + 'similarity_histogram.png')
    plt.show()
