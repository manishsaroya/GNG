"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
Evaluates the roadmaps
"""
from persistence.utils import load_graph, load_hilbert_map, convert_map_dict_to_array
from prm import hilbert_samples, collision_check
from future.utils import iteritems
import networkx as nx
import numpy as np
import heapq
import matplotlib.pyplot as pl
import pickle
from math import sqrt
from sklearn.neighbors import NearestNeighbors


def euclidean_distance(a, b):
    """Calculate distance between two points."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def determine_nclosest_vertices(graph, curnode, n):
    """Where this curnode is actually the x,y index of the data we want to analyze."""
    pos = nx.get_node_attributes(graph, 'pos')
    templist = []
    for node, position in iteritems(pos):
        dist = euclidean_distance(curnode, position)
        templist.append([node, dist])
    distlist = np.array(templist)

    ind = np.lexsort((distlist[:, 0], distlist[:, 1]))
    distlist = distlist[ind]
    return distlist[:n]

def save_img(data, graph,start_sample, start, goal_sample, goal, path_nodes, dir, fig_num=1, save_data=True, save_graph=True):
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
    # for node_1, node_2 in graph.edges:
    #     weights = np.concatenate([[position[node_1]], [position[node_2]]])
    #     line, = pl.plot(*weights.T, color='lightsteelblue')
    #     pl.setp(line, linewidth=2, color='lightsteelblue')
    # pl.title('Test image')

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


def calculate_distances(graph, starting_vertex, goal_vertex):
    position = nx.get_node_attributes(graph, 'pos')
    distances = {vertex: float('infinity') for vertex in graph.nodes()}
    prev = {vertex: None for vertex in graph.nodes()}
    distances[starting_vertex] = 0
    nodes_explored = 0
    pq = [(0, starting_vertex)]
    while len(pq) > 0:
        current_distance, current_vertex = heapq.heappop(pq)
        nodes_explored +=1
        if current_vertex==goal_vertex:
            return distances, prev, nodes_explored
        # Nodes can get added to the priority queue multiple times. We only
        # process a vertex the first time we remove it from the priority queue.
        if current_distance > distances[current_vertex]:
            continue

        all_nbrs = nx.all_neighbors(graph, current_vertex)
        for neighbor in all_nbrs:
            distance = current_distance + euclidean_distance(position[current_vertex], position[neighbor])

            # Only consider this new path if it's better than any path we've
            # already found.
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                prev[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))

    return distances, prev, nodes_explored

def convert_gng_to_nxgng(path,map_array, obs_threshold, resolution):
    with open(path, 'rb') as tf:
        g = pickle.load(tf)
    nxgraph = nx.Graph()
    nodeid = {}
    for indx, node in enumerate(g.graph.nodes):
        nodeid[node] = indx
        nxgraph.add_node(nodeid[node], pos=(node.weight[0][0], node.weight[0][1]))
    positions = nx.get_node_attributes(nxgraph, "pos")
    for node_1, node_2 in g.graph.edges:
        if collision_check(map_array, positions[nodeid[node_1]],
                                           positions[nodeid[node_2]], obs_threshold, resolution):
            nxgraph.add_edge(nodeid[node_1], nodeid[node_2])
    return nxgraph


def add_start_n_goal_to_graph(nxgraph, start_loc, goal_loc, save_pickle, map_array_):
    positions = nx.get_node_attributes(nxgraph, "pos")

    start_node_list = determine_nclosest_vertices(nxgraph, start_loc, 7-1)
    goal_node_list = determine_nclosest_vertices(nxgraph, goal_loc, 7-1)

    # adding start and goal to the graph
    nxgraph.add_node(len(nxgraph.nodes), pos=start_loc)
    nxgraph.add_node(len(nxgraph.nodes), pos=goal_loc)
    # Check for collision in start and goal
    for i in range(len(start_node_list)):
        start_collision_check = collision_check(map_array_, positions[start_node_list[i][0]], start_loc,
                                                obstacle_threshold, resolution)
        if start_collision_check and euclidean_distance(positions[start_node_list[i][0]], start_loc) < 5.0:
            nxgraph.add_edge(start_node_list[i][0], len(nxgraph.nodes) - 2)
            if not save_pickle:
                print("success with connecting start", start_loc)

    for i in range(len(goal_node_list)):
        goal_collision_check = collision_check(map_array_, positions[goal_node_list[i][0]], goal_loc,
                                               obstacle_threshold, resolution)
        if goal_collision_check and euclidean_distance(positions[goal_node_list[i][0]], goal_loc) < 5.0:
            nxgraph.add_edge(goal_node_list[i][0], len(nxgraph.nodes) - 1)
            if not save_pickle:
                print("success with connecting goal", goal_loc)
    return nxgraph

#
# def start_goal_addition(nxgraph, start_loc, goal_loc, save_pickle):
#     sample_list = [start_loc, goal_loc]
#     positions =
#     # find k nearest neighbor
#     nbrs = NearestNeighbors(n_neighbors=args.k_nearest, algorithm='ball_tree').fit(sample_list)
#     distances, indices = nbrs.kneighbors(sample_list)
#     # add graph nodes
#     for indx, s in enumerate(sample_list):
#         nxgraph.add_node(indx, pos=(s[0], s[1]))
#     # add graph edges
#     for row, node_adjacency_list in enumerate(indices):
#         for column, other_node in enumerate(node_adjacency_list):
#             distance_metric = distances[row][column] < args.connection_radius
#             collision_metric = collision_check(map_array, sample_list[node_adjacency_list[0]],
#                                                sample_list[other_node], args.obstacle_threshold, resolution)
#             validation_metric = node_adjacency_list[0] != other_node
#             if distance_metric and collision_metric and validation_metric:
#                 prm_graph.add_edge(node_adjacency_list[0], other_node, distance=distances[row][column])
#     print("rendering")


if __name__ == "__main__":
    exp_factor = 30
    used_stored_samples = True
    save_pickle = True
    test_list = [27,
                 77,
                 95,
                 97,
                 103,
                 118,
                 119,
                 121,
                 131,
                 168,
                 193,
                 195,
                 235,
                 241,
                 259,
                 260,
                 261,
                 270,
                 284,
                 305,
                 318,
                 343,
                 358,
                 360,
                 368,
                 382,
                 386,
                 394,
                 395,
                 431,
                 452,
                 455,
                 456]
    obstacle_threshold = 0.4

    # TODO: Finalize map to be used as of now using new map
    # load map
    map_data, resolution = load_hilbert_map(map_type="intel")
    map_array = convert_map_dict_to_array(map_data, resolution)
    # load graph
    with open("ground_map_q_resolution.pickle", 'rb') as tf:
        ground_map_data = pickle.load(tf)
    ground_resolution = 0.3
    ground_map_array = convert_map_dict_to_array(ground_map_data, ground_resolution)
    # roadmap_types = ["gng", "gng_top", "prm", "prm_dense"]
    roadmap_types = ["gng_top", "gng", "prm_dense"]
    data_save_dic = {"gng": "gng_output/", "gng_top": "gng_top_output/", "prm": "prm_output/",
                     "prm_dense": "prm_dense_output/"}

    gng_path = "../persistence/output/exp_factor-30-max_epoch-300-max_edge_age-20-date-2020-07-16-09-26-03/gng300.pickle"
    prm_path = "output/max_nodes-1208-k_nearest-5-connection_radius-5.0-date-2020-06-25-11-57-59/prm.pickle"
    gng_top_path = "../persistence/output/exp_factor-30-max_epoch-300-max_edge_age-20-date-2020-07-16-09-57-08/gng300.pickle"
    prm_dense_path = "output/max_nodes-4000-k_nearest-7-connection_radius-5.0-date-2020-07-23-00-36-05/prm.pickle"

    with open("test_samples/test_data1.pickle", 'rb') as tf:
        test_data = pickle.load(tf)
    goal_list = test_data[0]
    start_list = test_data[1]

    for roadmap in roadmap_types:

        if roadmap == "gng":
            prm_graph = convert_gng_to_nxgng(gng_path, map_array, obstacle_threshold, resolution)
        elif roadmap == "gng_top":
            prm_graph = convert_gng_to_nxgng(gng_top_path, map_array, obstacle_threshold, resolution)
        elif roadmap == "prm":
            prm_graph = load_graph(prm_path)
        else:  # roadmap == "prm_dense":
            prm_graph = load_graph(prm_dense_path)

        success_list = []
        node_explored_list = []
        distance_to_goal_list = []
        if save_pickle:
            eval_iterator = range(len(goal_list))
        else:
            eval_iterator = test_list

        for lamda_ in eval_iterator:
            goal_loc = goal_list[lamda_]
            start_loc = start_list[lamda_]
            if roadmap == "gng" or roadmap == "gng_top":
                full_graph = add_start_n_goal_to_graph(prm_graph.copy(), start_loc, goal_loc, save_pickle, map_array)
            else:
                full_graph = add_start_n_goal_to_graph(prm_graph.copy(), start_loc, goal_loc, save_pickle, ground_map_array)

            start_node = len(full_graph.nodes) - 2
            goal_node = len(full_graph.nodes) - 1
            kl, prev, nodes_explored = calculate_distances(full_graph, start_node, goal_node)
            path_exists = kl[goal_node] != float('infinity')
            path_nodes = []
            if path_exists:
                success_list.append(True)
                pointer = goal_node
                path_nodes.append(goal_node)
                while pointer != start_node:
                    path_nodes.append(prev[pointer])
                    pointer = prev[pointer]
                node_explored_list.append(nodes_explored)
                distance_to_goal_list.append(kl[goal_node])
            else:
                success_list.append(False)
                node_explored_list.append(None)
                distance_to_goal_list.append(None)

            if not save_pickle:
                save_img(map_data, full_graph, start_loc, start_node, goal_loc, goal_node,
                       path_nodes, data_save_dic[roadmap], fig_num=lamda_, save_graph=False)

        print("############## for ", roadmap)
        print("success trial", np.sum(success_list))
        print("success_list:", success_list)
        print("nodes explored list",node_explored_list)
        print("distance to goal list", distance_to_goal_list)
        if save_pickle:
            if roadmap == "gng":
                with open(data_save_dic[roadmap] + 'gng1208_200.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "gng_top":
                with open(data_save_dic[roadmap] + 'gngtop_1208_200.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "prm":
                with open(data_save_dic[roadmap] + 'prm1208.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "prm_dense":
                with open(data_save_dic[roadmap] + 'prmdense_2500.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)

