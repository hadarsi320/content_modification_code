import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    similarity_dir = '/lv_local/home/hadarsi/pycharm_projects/content_modification_code/output/run_26_9' \
                     '/similarity_results/'
    plots_dir = './plots/'
    similarity_files = sorted(os.listdir(similarity_dir))

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    sim_lists = []
    for file in similarity_files:
        sim_list = []
        with open(similarity_dir + file) as f:
            for line in f:
                value = float(line.split()[1])
                if value == 0:
                    break
                sim_list.append(value)
        sim_lists.append(sim_list)

    rounds = max(len(sim_list) for sim_list in sim_lists)
    sim_matrix = np.array([list for list in sim_lists if len(list) == rounds])

    # line plot
    averaged_mat = np.average(sim_matrix, axis=0)
    plt.plot(range(1, rounds+1), averaged_mat, 'o-')
    plt.xticks(range(1, rounds+1))
    plt.title(f'Similarity Measure Averaged Across {len(sim_matrix)} Competitions')
    plt.savefig(plots_dir + 'similarity_plots.png')
    plt.xlabel('Round')
    plt.ylabel('Cosine Similarity')
    plt.show()

    # histogram
    # alpha = 0.5
    # bins = 12
    # plt.hist(sim_matrix[:, 0], bins=bins, label='First Round', alpha=alpha)
    # for i in range(0, sim_matrix.shape[1], 10):
    #     if i == 0 or i == sim_matrix.shape[1] - 1:
    #         continue
    #     plt.hist(sim_matrix[:, i], bins=bins, alpha=alpha, label='Round {}'.format(i))
    # plt.hist(sim_matrix[:, -1], bins=bins, label='Last Round', alpha=alpha)
    # plt.legend()
    # plt.title('Similarity Histogram')
    # plt.xlabel('Cosine Similarity')
    # plt.ylabel('Counts')
    # plt.savefig(plots_dir + 'similarity_histogram.png')
    # plt.show()
