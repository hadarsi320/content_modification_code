import numpy as np
import os
import matplotlib.pyplot as plt


def plot(data, start=0, stop=None, shape='o-', title=None, xlabel=None, ylabel=None, save_file=None):
    if not stop:
        stop = len(data) + start
    plt.plot(range(start, stop), data, shape)
    plt.xticks(range(start, stop))
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if save_file:
        plt.savefig(save_file)
    plt.show()


if __name__ == '__main__':
    similarity_dir = './output/run_10_1/similarity_results'
    plots_dir = './plots/'
    similarity_files = sorted(os.listdir(similarity_dir))

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    sim_lists = [], []
    for file in similarity_files:
        sim_list = [], []
        with open(similarity_dir + '/' + file) as f:
            for num, line in enumerate(f):
                if num == 0:
                    continue
                for i in [0, 1]:
                    sim_list[i].append(float(line.split('\t')[i + 1]))
        for i in [0, 1]:
            sim_lists[i].append(sim_list[i])

    rounds = max(len(list) for list in sim_lists[0])
    matrices = [np.array([list for list in sim_lists[i] if len(list) == rounds]) for i in range(2)]

    averaged_mat = np.average(matrices[0], axis=0)
    plot(averaged_mat, title=f'Lexical Similarity Measure Averaged Across {len(matrices[0])} competitions',
         xlabel='Round', ylabel='Cosine Similarity')

    averaged_mat = np.average(matrices[1], axis=0)
    plot(averaged_mat, title=f'Embedding Similarity Measure Averaged Across {len(matrices[1])} competitions',
         xlabel='Round', ylabel='Cosine Similarity')

    # histogram
    # alpha = 0.5
    # bins = 12
    # plt.hist(sim_matrix[:, 0], bins=bins, label='First Round', alpha=alpha)
    # for i in range(0, sim_matrix.shape[1], 4):
    #     if i == 0 or i == sim_matrix.shape[1] - 1:
    #         continue
    #     plt.hist(sim_matrix[:, i], bins=bins, alpha=alpha, label='Round {}'.format(i))
    # plt.hist(sim_matrix[:, -1], bins=bins, label='Last Round', alpha=alpha)
    # plt.legend()
    # plt.title('Similarity Histogram')
    # plt.xlabel('Cosine Similarity')
    # plt.ylabel('Counts')
    # # plt.savefig(plots_dir + 'similarity_histogram.png')
    # plt.show()
