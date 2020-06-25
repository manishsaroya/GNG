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


def euclidean_distance(a, b):
    """Calculate distance between two points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def determine_2closest_vertices(graph, curnode):
    """Where this curnode is actually the x,y index of the data we want to analyze."""
    pos = nx.get_node_attributes(graph, 'pos')
    templist = []
    for node, position in iteritems(pos):
        dist = euclidean_distance(curnode, position)
        templist.append([node, dist])
    distlist = np.array(templist)

    ind = np.lexsort((distlist[:, 0], distlist[:, 1]))
    distlist = distlist[ind]

    return distlist[0], distlist[1]

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
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([[position[node_1]], [position[node_2]]])
        line, = pl.plot(*weights.T, color='lightsteelblue')
        pl.setp(line, linewidth=2, color='lightsteelblue')
    pl.title('Test image')
    #if samples is not None:
    pl.scatter(start_sample[0], start_sample[1], marker='v', facecolors='cyan')
    pl.scatter(goal_sample[0], goal_sample[1], marker='*', facecolors='cyan')
    pl.scatter(position[start][0], position[start][1], marker='^', facecolors='yellow')
    pl.scatter(position[goal][0], position[goal][1], marker='*', facecolors='yellow')

    for path_n in path_nodes:
        pl.scatter(position[path_n][0], position[path_n][1],s=90, marker='*', facecolors='white')

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

def convert_gng_to_nxgng(path,map_array, resolution):
    with open(path, 'rb') as tf:
        g = pickle.load(tf)
    nxgraph = nx.Graph()
    nodeid = {}
    for indx, node in enumerate(g.graph.nodes):
        nodeid[node] = indx
        nxgraph.add_node(nodeid[node], pos=(node.weight[0][0], node.weight[0][1]))
        #print(node)
        #print(indx)
        #print(node.weight)
    positions = nx.get_node_attributes(nxgraph, "pos")
    for node_1, node_2 in g.graph.edges:
        #collision_metric = collision_check(map_array, sample_list[node_adjacency_list[0]],
        #                                   sample_list[other_node], args.obstacle_threshold)
        if collision_check(map_array, positions[nodeid[node_1]],
                                           positions[nodeid[node_2]], 0.4, resolution):
            nxgraph.add_edge(nodeid[node_1], nodeid[node_2])
    return nxgraph



if __name__ =="__main__":
    # load map
    map_data, resolution = load_hilbert_map(map_type="intel")
    map_array = convert_map_dict_to_array(map_data, resolution)
    # load graph

    gng_bool = False
    prm_bool = False
    prm_dense_bool = False
    gng_top_bool = True

    gng_path = "../persistence/output/exp_factor-30-max_epoch-200-max_edge_age-20-date-2020-06-21-01-27-07/gng200.pickle"
    #prm_path = "output/max_nodes-2000-k_nearest-5-connection_radius-5.0-date-2020-06-22-01-34-03/prm.pickle"
    prm_path = "output/max_nodes-1208-k_nearest-5-connection_radius-5.0-date-2020-06-25-11-57-59/prm.pickle"
    gng_top_path = "../persistence/output/exp_factor-30-max_epoch-200-max_edge_age-20-date-2020-06-25-12-36-25/gng200.pickle"
    prm_dense_path = "output/max_nodes-2500-k_nearest-5-connection_radius-5.0-date-2020-06-25-12-19-34/prm.pickle"
    #prm_graph = load_graph(prm_path)
    if gng_bool:
        prm_graph = convert_gng_to_nxgng(gng_path, map_array, resolution)
    elif prm_bool:
        prm_graph = load_graph(prm_path)
    elif prm_dense_bool:
        prm_graph = load_graph(prm_dense_path)
    elif gng_top_bool:
        prm_graph = convert_gng_to_nxgng(gng_top_path, map_array, resolution)

    exp_factor = 30

    # prm_graph
    # num_test_query = 1
    # get start from hilbert maps
    #goal_loc = [[11.2, -21.1]] #hilbert_samples(map_data.copy(),30, num_samples=1)
    #start_loc = [[9.1, -18.7]] #hilbert_samples(map_data.copy(),30, num_samples=1)

    #goal_list = hilbert_samples(map_data.copy(), 30, num_samples=500)
    #start_list = hilbert_samples(map_data.copy(), 30, num_samples=500)

    with open("test_samples/test_data.pickle", 'rb') as tf:
       test_data = pickle.load(tf)
    goal_list = test_data[0]
    start_list = test_data[1]
    # with open("test_samples/" + 'test_data.pickle', 'wb') as handle:
    #     pickle.dump([goal_list,start_list], handle)
    success_list = []
    node_explored_list = []
    distance_to_goal_list = []
    for lamda_ in range(len(goal_list)):
        goal_loc = [goal_list[lamda_]]  #hilbert_samples(map_data.copy(),30, num_samples=1)
        start_loc = [start_list[lamda_]] #hilbert_samples(map_data.copy(),30, num_samples=1)
        start_node, _ = determine_2closest_vertices(prm_graph,start_loc[0])
        goal_node, _ = determine_2closest_vertices(prm_graph,goal_loc[0])
        #save_img(map_data, prm_graph, start_loc[0], int(start_node[0]), goal_loc[0], int(goal_node[0]), "", save_graph=False)
        # goal_loc = sample_list
        kl, prev, nodes_explored =calculate_distances(prm_graph,int(start_node[0]), int(goal_node[0]))
        path_exists = kl[goal_node[0]] != float('infinity')
        if path_exists:
            success_list.append(True)
            path_nodes = []
            pointer = goal_node[0]
            path_nodes.append(int(goal_node[0]))
            while pointer!= start_node[0]:
                path_nodes.append(prev[pointer])
                pointer = prev[pointer]
            #print(path_nodes)
            #print(kl[goal_node[0]])
            #print("nodes explored",nodes_explored)
            node_explored_list.append(nodes_explored)
            #print("path cost", kl[goal_node[0]] + start_node[1] + goal_node[1])
            distance_to_goal_list.append(kl[goal_node[0]] + start_node[1] + goal_node[1])
            #save_img(map_data, prm_graph, start_loc[0], int(start_node[0]), goal_loc[0], int(goal_node[0]), path_nodes, "", fig_num=lamda_,save_graph=False)
        else:
            success_list.append(False)
            node_explored_list.append(None)
            distance_to_goal_list.append(None)
            #print("No path")

    print("##############")
    print("success trial", np.sum(success_list))
    print("success_list:", success_list)
    print("nodes explored list",node_explored_list)
    print("distance to goal list", distance_to_goal_list)
    if gng_bool:
        with open("test_output/" + 'gng1208_200.pickle', 'wb') as handle:
            pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
    elif gng_top_bool:
        with open("test_output/" + 'gngtop_1208_200.pickle', 'wb') as handle:
            pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
    elif prm_bool:
        with open("test_output/" + 'prm1208.pickle', 'wb') as handle:
            pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
    elif prm_dense_bool:
        with open("test_output/" + 'prmdense_2500.pickle', 'wb') as handle:
            pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)

