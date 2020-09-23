import numpy as np
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    similarity_dir = './similarity_results/'
    similarity_files = sorted(os.listdir(similarity_dir))

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
    for sim_list in sim_lists:
        if len(sim_list) < rounds:
            sim_lists.remove(sim_list)

    sim_matrix = np.array(sim_lists)
    averaged_mat = np.average(sim_matrix, axis=0)
    plt.plot(range(1, rounds+1), averaged_mat)
    plt.xticks(range(1, rounds+1))
    plt.title(f'Similarity measure averaged across {len(sim_matrix)} competitions')
    plt.show()

