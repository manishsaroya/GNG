"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""

import pickle
import matplotlib.pyplot as plt


def save_img(data, graph,start_sample, start, goal_sample, goal, path_nodes, dir="", fig_num=1, save_data=True, save_graph=True):
    """
    :param data: map data dictionary
    :param graph: prm nx graph
    :param dir: output directory path
    :return: save prm graph and visualization image to output directory
    """
    fig = pl.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    pl.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap='jet', s=70, vmin=0, vmax=1, edgecolors='')
    pl.colorbar()
    position = nx.get_node_attributes(graph, 'pos')
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([[position[node_1]], [position[node_2]]])
        line, = pl.plot(*weights.T, color='lightsteelblue')
        pl.setp(line, linewidth=2, color='lightsteelblue')
    pl.title('Test image')

    pl.scatter(start_sample[0], start_sample[1],s=90, marker='v', facecolors='cyan')
    pl.scatter(goal_sample[0], goal_sample[1],s=90, marker='*', facecolors='cyan')
    pl.scatter(position[start][0], position[start][1], s=90, marker='v', facecolors='yellow')
    pl.scatter(position[goal][0], position[goal][1], s=90, marker='*', facecolors='yellow')

    for path_n in path_nodes:
        pl.scatter(position[path_n][0], position[path_n][1],s=40, marker='*', facecolors='red')

    for i in range(len(path_nodes)-1):
        weights = np.concatenate([[position[path_nodes[i]]], [position[path_nodes[i+1]]]])
        line, = pl.plot(*weights.T, color='white')
        pl.setp(line, linewidth=2, color='white')


    if save_data:
        pl.savefig(dir + "prm{}.eps".format(fig_num))
    if save_graph:
        with open(dir + 'prm.pickle', 'wb') as handle:
            pickle.dump(graph, handle)


if __name__ == "__main__":
    # load map
    with open("../prm/ground_map_q_resolution.pickle", 'rb') as tf:
        data = pickle.load(tf)

    plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap='jet', s=70/4, vmin=0, vmax=1, edgecolors='')
    plt.colorbar()
    plt.show()