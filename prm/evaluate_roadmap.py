"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
Evaluates the roadmaps
"""
from persistence.utils import load_graph, load_hilbert_map
from prm import hilbert_samples
from future.utils import iteritems
import networkx as nx
import numpy as np
import heapq
import matplotlib.pyplot as pl


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

def save_img(data, graph,start_sample, start, goal_sample, goal, path_nodes, dir, save_data=True, save_graph=True):
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
        pl.savefig(dir + "prm.eps")
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


if __name__ =="__main__":
    # load graph
    prm_path = "output/max_nodes-2000-k_nearest-5-connection_radius-5.0-date-2020-06-22-01-34-03/prm.pickle"
    prm_graph = load_graph(prm_path)
    exp_factor = 30
    # load map
    map_data, resolution = load_hilbert_map(map_type="intel")
    # prm_graph
    # num_test_query = 1
    # get start from hilbert maps
    goal_loc = [[11.2, -21.1]] #hilbert_samples(map_data.copy(),30, num_samples=1)
    start_loc = [[9.1, -18.7]] #hilbert_samples(map_data.copy(),30, num_samples=1)
    start_node, _ = determine_2closest_vertices(prm_graph,start_loc[0])
    goal_node, _ = determine_2closest_vertices(prm_graph,goal_loc[0])
    #save_img(map_data, prm_graph, start_loc[0], int(start_node[0]), goal_loc[0], int(goal_node[0]), "", save_graph=False)
    # goal_loc = sample_list
    kl, prev, nodes_explored =calculate_distances(prm_graph,int(start_node[0]), int(goal_node[0]))
    path_nodes = []
    pointer = goal_node[0]
    path_nodes.append(int(goal_node[0]))
    while pointer!= start_node[0]:
        path_nodes.append(prev[pointer])
        pointer = prev[pointer]
    print(path_nodes)
    print(kl[goal_node[0]])
    print("nodes explored",nodes_explored)
    save_img(map_data, prm_graph, start_loc[0], int(start_node[0]), goal_loc[0], int(goal_node[0]), path_nodes, "", save_graph=False)


