"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
# get the map
from persistence.biased_sampling import get_samples, samples_plot
from persistence.utils import load_hilbert_map, convert_map_dict_to_array, convert_gng_to_nxgng
from persistence.drive_hilbert_persistence import get_top_n_persistence_node_location
from persistence.utils import get_topological_accuracy, get_0hom_topological_accuracy
import numpy as np
from neupy import algorithms, utils
import matplotlib.pyplot as plt
import os
import argparse
import datetime
import pickle
import networkx as nx
import time


def create_gng(max_nodes, step=0.2, n_start_nodes=2, max_edge_age=50):
    return algorithms.GrowingNeuralGas(
        n_inputs=2,
        n_start_nodes=n_start_nodes,

        shuffle_data=True,
        verbose=True,

        step=step,
        neighbour_step=0.005,

        max_edge_age=max_edge_age,
        max_nodes=max_nodes,

        n_iter_before_neuron_added=100,
        after_split_error_decay_rate=0.5,
        error_decay_rate=0.995,
        min_distance_for_update=0.01,
    )

def save_img(data, dir, fig_num=1, save_data=True, save_graph=True, persistence_birthnode=None, persistence_1homnode=None, figure_cmap="jet", map_type="intel"):
    """
    :param data: map data dictionary
    :param graph: prm nx graph
    :param dir: output directory path
    :return: save prm graph and visualization image to output directory
    """

    # if map_type=="intel":
    #     fig = plt.figure(figsize=(40/4, 35/4))
    #     ax = fig.add_subplot(111)
    #     plt.axis("equal")
    #     plt.xlim(-11, 18)
    #     plt.ylim(-23.5, 5.8)
    #     plt.style.use('seaborn-dark')
    #     plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], marker="s", s=37, vmin=0, vmax=1, edgecolors='')
    # elif map_type=="fhw":
    #     fig = pl.figure(figsize=(8, 3))
    #     ax = fig.add_subplot(111)
    #     pl.axis("equal")
    #     pl.ylim(-6, 23)
    #     pl.xlim(-32, 50)
    #     pl.scatter(data['Xq'][:, 1], data['Xq'][:, 0], c=data['yq'], cmap=figure_cmap, s=11, vmin=0, vmax=1,\
    #                edgecolors='')
    # else:
    #     fig = pl.figure(figsize=(8, 3))
    #     ax = fig.add_subplot(111)
    #     pl.axis("equal")
    #
    #     pl.xlim(-25.5, 17)
    #     pl.ylim((-10, 10))
    #     pl.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap=figure_cmap, s=11, vmin=0, vmax=1, edgecolors='')
    # cbar = plt.colorbar(fraction= 0.067, pad=0.02)
    # cbar.set_label(label="Occupancy probability", size=25)
    # #cbar.set_label(label="Occupancy probability", size=15)
    # cbar.ax.tick_params(labelsize=13)
    # plt.xticks([],fontsize=11)
    # plt.yticks([], fontsize=11)
    #
    # if persistence_birthnode is not None:
    #     for i in range(len(persistence_birthnode)):
    #         plt.plot(persistence_birthnode[i][0], persistence_birthnode[i][1], "o", markersize=20, markerfacecolor="None", markeredgecolor="blue", markeredgewidth=3.5)
    #
    # if persistence_1homnode is not None:
    #     for i in range(len(persistence_1homnode)):
    #         plt.plot(persistence_1homnode[i][0], persistence_1homnode[i][1], "D", markersize=20, markerfacecolor="None", markeredgecolor="red", markeredgewidth=3.5)
    # fig.tight_layout()
    # if save_data:
    #     plt.savefig(dir + "{}_prm{}.png".format(map_type, fig_num), dpi=300)


    #plt.gcf().clear()
    #plt.clf()
    z = persistence_birthnode
    f = persistence_1homnode
    fig, ax = plt.subplots(figsize=(40 / 12, 40 / 12))
    plt.axis("equal")
    #plt.style.use('seaborn-dark')
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    x = np.linspace(-0.02, 1.02, 1000)
    ax.plot(x, x, color="k")
    if len(z) > 0:
        ax.plot(z[:, 0], z[:, 1], 'o', markerfacecolor="None", markeredgecolor="blue", markeredgewidth=1.5, label="0-dim")
    if len(f) > 0:
        ax.plot(f[:, 0], f[:, 1], 'D', markerfacecolor="None", markeredgecolor="red", markeredgewidth=1.5, label="1-dim")
    ax.plot([0.25, 0.25], [-0.02, 1.02], "r--",linewidth=1)
    ax.plot([-0.02, 1.02], [0.25, 0.25], "b--", linewidth=1)
    ax.legend(loc="lower right")
    ax.set_yticks(np.arange(0, 2, step=1))
    ax.set_xticks(np.arange(0, 2, step=1))
    # ax[1].set_title("output PersistenceDgm")
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.xaxis.labelpad = -9
    ax.yaxis.labelpad = -9
    fig.tight_layout()
    if not os.path.exists("./persistence_new"):
        os.makedirs("./persistence_new")
    plt.savefig(dir + 'persistence_dgm.png', dpi=300)



def draw_image(data,resolution, dir, fignum, graph=None, persistence_birthnode=None, persistence_1homnode=None, samples=None, show=True):
    fig = plt.figure(figsize=(40/4, 35/4))
    plt.axis("equal")

    #plt.style.use('seaborn-dark')
    plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap="jet", s=40, vmin=0, vmax=1, edgecolors='')
    #plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], s=10, vmin=0, vmax=1, edgecolors='')
    plt.colorbar(fraction= 0.047, pad=0.02)
    if graph is not None:
        for node_1, node_2 in graph.edges:
            weights = np.concatenate([node_1.weight, node_2.weight])
            line, = plt.plot(*weights.T, color='lightsteelblue')
            plt.setp(line, linewidth=2, color='lightsteelblue')

    if samples is not None:
        plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1], marker='v', facecolors='cyan')

    if persistence_birthnode is not None:
        for i in range(len(persistence_birthnode)):
            plt.plot(persistence_birthnode[i][0], persistence_birthnode[i][1], "*", markersize=15, markerfacecolor="None", markeredgecolor="blue", markeredgewidth=1.5)

    if persistence_1homnode is not None:
        for i in range(len(persistence_1homnode)):
            plt.plot(persistence_1homnode[i][0], persistence_1homnode[i][1], "*", markersize=15, markerfacecolor="None", markeredgecolor="red", markeredgewidth=1.5)

    if show:
        plt.savefig(dir + "graph{}.eps".format(fignum))
    # plt.show()


def normalize(data, obs_threshold, exp_factor=20):
    # toggle the probabilities
    data['yq'] = np.ones(len(data['yq'])) - data['yq']
    data["yq"][data["yq"] < (1-obs_threshold)] = 0
    #data['yq'] = np.exp(exp_factor * data['yq'])
    # normalize the probabilities
    data['yq'] /= np.linalg.norm(data['yq'], ord=1)
    return data


def plot_loss(train_error_mean, train_error_std, dir):
    plt.figure(figsize=(5, 5))
    plt.plot(train_error_mean, label="train error", color="g")
    plt.fill_between(range(len(train_error_mean)), np.array(train_error_mean) - np.array(train_error_std),
                     np.array(train_error_mean) + np.array(train_error_std), \
                     color="g", label="std", alpha=0.1)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    # plt.xlim(0, 250)
    # plt.yticks(np.arange(60, 110, step=10))
    plt.title("Train Error")
    plt.savefig(dir + "training_error.png")


def get_multi_gauss_samples(local_map_data, persistence_loc, exp_factor, obs_threshold, std_dev_):
    sample_list = []
    for i in range(2):
        small_list = get_samples(local_map_data.copy(),
                                 persistence_loc[np.random.randint(0, len(persistence_loc))], exp_factor, obs_threshold,
                                 scale=std_dev_, num_samples=int(600 / 2))
        sample_list.extend(small_list)
    return sample_list

def get_important_regions(g, regions):
    nxgraph = nx.Graph()
    nodeid = {}
    for indx, node in enumerate(g.graph.nodes):
        nodeid[node] = indx
        nxgraph.add_node(nodeid[node], pos=(node.weight[0][0], node.weight[0][1]))
    for node_1, node_2 in g.graph.edges:
        nxgraph.add_edge(nodeid[node_1], nodeid[node_2])

    for clique in nx.enumerate_all_cliques(nxgraph):
        print(clique)

    print(regions)
    fig = plt.figure(figsize=(40 / 4, 35 / 4))
    plt.axis("equal")
    #plt.colorbar(fraction= 0.047, pad=0.02)
    position = nx.get_node_attributes(nxgraph, 'pos')
    if nxgraph is not None:
        for clique in nx.find_cliques(nxgraph):
            if len(clique)>2:
                for i in range(len(clique)):
                    weights = np.concatenate([[position[clique[i]]], [position[clique[i-1]]]])
                    line, = plt.plot(*weights.T, color='lightsteelblue')
                    plt.setp(line, linewidth=2, color='lightsteelblue')
    plt.show()
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--exp_factor', type=int, default=9)
    parser.add_argument('--max_edge_age', type=int, default=56)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--max_nodes', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='./output')
    parser.add_argument('--top_n_persistence', type=int, default=25)
    parser.add_argument('--local_distance', type=float, default=5.0)
    parser.add_argument('--local_distance_0hom', type=float, default=1.1)
    parser.add_argument('--is_bias_sampling', type=bool, default=True)
    parser.add_argument('--is_topology_feedback', type=bool, default=True)
    parser.add_argument('--bias_ratio', type=float, default=0.75)
    parser.add_argument('--obstacle_threshold', type=float, default=0.5)
    parser.add_argument('--map_type', type=str, default="freiburg")
    args = parser.parse_args()

    args.log_dir = './output/exp_factor-' + args.map_type + str(args.exp_factor) + "-is-topology-feedback-" + \
                   str(args.is_topology_feedback) + "-is-bias-sampling-" + \
                   str(args.is_bias_sampling) + "-bias_ratio-" + str(args.bias_ratio) +"-max_epoch-" + \
                   str(args.max_epoch) + "-max_edge_age-" + str(args.max_edge_age) + "-date-" + \
                   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '/'

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tic = time.time()
    data, resolution = load_hilbert_map(map_type=args.map_type)
    #data["yq"] = 1.0 * (data["yq"] > args.obstacle_threshold)
    map_array = convert_map_dict_to_array(data, resolution)

    #exit()
    persistence_birth_nodes, persistence_weights = get_top_n_persistence_node_location(args.top_n_persistence, args.map_type, args.obstacle_threshold,
                                                                  location_type="death", feature_type=0)
    persistence_1hom_nodes, persistence_1hom_weights = get_top_n_persistence_node_location(args.top_n_persistence, args.map_type, args.obstacle_threshold,
                                                                  location_type="death", feature_type=1)
    #
    # with open('fhw_persistence.pickle', 'wb') as handle:
    #     pickle.dump([persistence_birth_nodes, persistence_weights, persistence_1hom_nodes, persistence_1hom_weights], handle)

    # with open('fhw_persistence.pickle', 'rb') as tf:
    # 	persistence_birth_nodes, persistence_weights, persistence_1hom_nodes, persistence_1hom_weights = pickle.load(tf)

    original_data = data.copy()
    data = normalize(data, args.obstacle_threshold, args.exp_factor)

    # save_img(original_data, args.log_dir, 2, persistence_birthnode=persistence_birth_nodes, \
    #           persistence_1homnode=persistence_1hom_nodes)
    # exit()
    # draw_image(original_data, resolution, args.log_dir, 2, persistence_birthnode=persistence_birth_nodes, \
    #            persistence_1homnode=persistence_1hom_nodes, show=True)
    # exit()
    # GNG learning code
    utils.reproducible()
    gng = create_gng(args.max_nodes, max_edge_age=args.max_edge_age)
    all_samples = []
    train_error_mean = []
    train_error_std = []
    for epoch in range(args.max_epoch + 1):
        if args.is_bias_sampling and epoch > 70:
            if np.random.uniform(0, 1) < args.bias_ratio:
                sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]
            else:
                ##print("biasing epoch", epoch)
                if np.random.uniform(0,1) > 0.5:
                    features = persistence_birth_nodes.copy()
                    if args.is_topology_feedback:
                        gng_nx = convert_gng_to_nxgng(gng, map_array, args.obstacle_threshold, resolution)
                        accuracy = get_0hom_topological_accuracy(gng_nx, features, args.local_distance_0hom)
                        accuracy_indices = [i for i, val in enumerate(accuracy) if val]
                        ##print("connected", accuracy_indices)
                        for i in sorted(accuracy_indices, reverse=True):
                            del features[i]

                    if len(features):
                        sample_list = get_multi_gauss_samples(original_data.copy(), features, args.exp_factor, args.obstacle_threshold, 1.2)
                        #samples_plot(original_data, sample_list, epoch, args.log_dir)
                    else:
                        ##print("ALL CONNECTIONS RESOLVED, UNIFORM SAMPLING FOR THIS EPOCH")
                        sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]
                    #sample_list = get_multi_gauss_samples(original_data.copy(), persistence_birth_nodes, args.exp_factor, 1.1)
                    #samples_plot(original_data, sample_list, epoch, args.log_dir)
                else:
                    features = persistence_1hom_nodes.copy()
                    if args.is_topology_feedback:
                        gng_nx = convert_gng_to_nxgng(gng, map_array, args.obstacle_threshold, resolution)
                        accuracy = get_topological_accuracy(gng_nx, features, args.local_distance)
                        accuracy_indices = [i for i, val in enumerate(accuracy) if val]
                        ##print(accuracy_indices)
                        for i in sorted(accuracy_indices, reverse=True):
                            del features[i]

                    # draw_image(original_data, resolution, args.log_dir, epoch, graph=gng.graph,
                    # 		   persistence_birthnode=persistence_birth_nodes, \
                    # 		   persistence_1homnode=persistence_1hom_nodes, show=True)
                    # get_important_regions(gng, persistence_1hom_nodes)

                    if len(features):
                        sample_list = get_multi_gauss_samples(original_data.copy(), features, args.exp_factor, args.obstacle_threshold, 2.5)
                        #samples_plot(original_data, sample_list, epoch, args.log_dir)
                    else:
                        print("ALL LOOPS RESOLVED, UNIFORM SAMPLING FOR THIS EPOCH")
                        sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]
        else:
            sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]

        #all_samples.extend(sample_list)
        gng.train(sample_list, epochs=1)
        train_error_mean.append(np.mean(gng.errors.train))
        train_error_std.append(np.std(gng.errors.train))
        if epoch % 10000 == 0:
            draw_image(original_data, resolution, args.log_dir, epoch, graph=gng.graph, persistence_birthnode=persistence_birth_nodes, \
                       persistence_1homnode=persistence_1hom_nodes, show=True)
            with open(args.log_dir + 'gng{:d}.pickle'.format(epoch), 'wb') as handle:
                pickle.dump(gng, handle)
    print("time elapsed", time.time()-tic)
    plot_loss(train_error_mean, train_error_std, args.log_dir)
    draw_image(original_data, resolution, args.log_dir, args.max_epoch, graph=gng.graph,
               persistence_birthnode=persistence_birth_nodes,
               persistence_1homnode=persistence_1hom_nodes, show=True)

