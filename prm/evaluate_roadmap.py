"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
Evaluates the roadmaps
"""
from persistence.utils import load_graph, load_hilbert_map, convert_map_dict_to_array, convert_gng_to_nxgng
from prm import collision_check
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


def plot_all_paths(data, path_set, graph):
    fig = pl.figure(figsize=(40/4, 35/4))
    ax = fig.add_subplot(111)
    pl.axis("equal")
    pl.ylim(-20, 10)
    pl.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap="jet", s=10, vmin=0, vmax=1, edgecolors='')
    pl.colorbar(fraction= 0.047, pad=0.02)
    position = nx.get_node_attributes(graph, 'pos')
    for path_nodes in path_set:
        for path_n in path_nodes:
            pl.scatter(position[path_n][0], position[path_n][1], s=40, marker='*', facecolors='red')
        #if len(path_nodes)>0:
        #    for i in range(len(path_nodes)-1):
        #        weights = np.concatenate([[position[path_nodes[i]]], [position[path_nodes[i+1]]]])
        #        line, = pl.plot(*weights.T, color='white')
        #        pl.setp(line, linewidth=3, color='white')
    pl.show()

def save_img(data, graph,start_sample, start, goal_sample, goal, path_nodes, dir, fig_num=1, save_data=True, save_graph=True, figure_cmap="jet"):
    """
    :param data: map data dictionary
    :param graph: prm nx graph
    :param dir: output directory path
    :return: save prm graph and visualization image to output directory
    """
    fig = pl.figure(figsize=(40/4, 35/4))
    ax = fig.add_subplot(111)
    pl.axis("equal")
    #pl.ylim(-20, 10)
    pl.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap=figure_cmap, s=10, vmin=0, vmax=1, edgecolors='')
    pl.colorbar(fraction= 0.047, pad=0.02)
    position = nx.get_node_attributes(graph, 'pos')
    # Plot the graph
    for node_1, node_2 in graph.edges:
        weights = np.concatenate([[position[node_1]], [position[node_2]]])
        line, = pl.plot(*weights.T, color='lightsteelblue')
        pl.setp(line, linewidth=2, color='lightsteelblue')

    # plot start and goal
    pl.scatter(start_sample[0], start_sample[1],s=90, marker='v', facecolors='cyan')
    pl.scatter(goal_sample[0], goal_sample[1],s=90, marker='*', facecolors='cyan')
    pl.scatter(position[start][0], position[start][1], s=90, marker='v', facecolors='yellow')
    pl.scatter(position[goal][0], position[goal][1], s=90, marker='*', facecolors='yellow')

    # plot path
    for path_n in path_nodes:
        pl.scatter(position[path_n][0], position[path_n][1],s=40, marker='*', facecolors='red')

    for i in range(len(path_nodes)-1):
        weights = np.concatenate([[position[path_nodes[i]]], [position[path_nodes[i+1]]]])
        line, = pl.plot(*weights.T, color='white')
        pl.setp(line, linewidth=3, color='white')


    if save_data:
        pl.savefig(dir + "freiburg_prm{}.eps".format(fig_num))
    if save_graph:
        with open(dir + 'freiburg_prm.pickle', 'wb') as handle:
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


def filter_graph_with_ground_truth_map(nxgraph, ground_truth_array, obs_threshold, ground_resolution):
    position = nx.get_node_attributes(nxgraph, 'pos')
    remove_count = 0
    all_edges = list(nxgraph.edges())
    for node_1, node_2 in all_edges:
        if not collision_check(ground_truth_array, position[node_1], position[node_2], obs_threshold, ground_resolution):
            nxgraph.remove_edge(node_1, node_2)
            remove_count += 1
    #print(remove_count)
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



if __name__ == "__main__":
    #exp_factor = 30
    used_stored_samples = True
    save_pickle = True
    test_list = [4] #, 66, 95, 147, 198, 260, 265, 290, 331, 341, 349, 354, 377, 394] #[131, 305, 358, 386, 394, 456] #[202, 332, 367]
    obstacle_threshold = 0.25

    # TODO: Finalize map to be used as of now using new map
    # load map
    map_data, resolution = load_hilbert_map(map_type="intel")
    map_array = convert_map_dict_to_array(map_data, resolution)

    ground_map_data = map_data.copy()
    ground_map_data["yq"] = 1.0 * (ground_map_data["yq"] > obstacle_threshold)

    roadmap_types = ["gng", "gng_top", "gng_top_feedback", "prm", "prm_dense_hilbert", "prm_dense"]
    #roadmap_types = [ "prm", "prm_dense_hilbert", "prm_dense"]
    #roadmap_types = ["gng", "gng_top", "gng_top_feedback"]
    #roadmap_types = ["gng_top", "gng", "gng_top_feedback"]
    data_save_dic = {"gng": "gng_output/", "gng_top": "gng_top_output/", "gng_top_feedback": "gng_top_feedback_output/", "prm": "prm_output/",
                     "prm_dense": "prm_dense_output/", "prm_dense_hilbert": "prm_dense_hilbert_output/"}

    """
    # Freiburgh Map
    # Map points: test_samples/freiburg_hilbert_maptest1.pickle
    prm_dense_path = "output/max_nodes-freiburg4000-obs-thres0.5-k_nearest-7-connection_radius-5.0-date-2020-10-11-00-31-27/prm.pickle"
    prm_dense_hilbert_path = "output/max_nodes-freiburg2000-obs-thres0.5-k_nearest-7-connection_radius-5.0-date-2020-10-11-00-36-45/prm.pickle"
    prm_path = "output/max_nodes-freiburg1000-obs-thres0.5-k_nearest-7-connection_radius-5.0-date-2020-10-11-00-30-17/prm.pickle"
    # Param 1
    # gng_top_path = "../persistence/output/exp_factor-freiburg9-is-topology-feedback-False-is-bias-sampling-True-bias_ratio-0.78-max_epoch-300-max_edge_age-60-date-2020-10-10-22-34-58/gng300.pickle"
    # gng_path = "../persistence/output/exp_factor-freiburg9-is-topology-feedback-False-is-bias-sampling-False-bias_ratio-0.78-max_epoch-300-max_edge_age-60-date-2020-10-10-22-19-05/gng300.pickle"
    # gng_top_feedback_path = "../persistence/output/exp_factor-freiburg9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.78-max_epoch-400-max_edge_age-60-date-2020-10-10-21-35-48/gng300.pickle"
    # Param 2
    gng_top_path = "../persistence/output/exp_factor-freiburg9-is-topology-feedback-False-is-bias-sampling-True-bias_ratio-0.75-max_epoch-300-max_edge_age-56-date-2020-10-10-23-41-43/gng300.pickle"
    gng_path = "../persistence/output/exp_factor-freiburg9-is-topology-feedback-False-is-bias-sampling-False-bias_ratio-0.75-max_epoch-300-max_edge_age-56-date-2020-10-10-23-45-21/gng300.pickle"
    gng_top_feedback_path = "../persistence/output/exp_factor-freiburg9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.75-max_epoch-300-max_edge_age-56-date-2020-10-10-23-37-29/gng300.pickle"
    """
    """
    #FHW Map
    # Map query points saved in fhw.pickle
    prm_dense_path = "output/max_nodes-fhw4000-obs-thres0.45-k_nearest-7-connection_radius-5.0-date-2020-10-11-02-38-06/prm.pickle"
    prm_dense_hilbert_path = "output/max_nodes-fhw2000-obs-thres0.45-k_nearest-7-connection_radius-5.0-date-2020-10-11-02-51-45/prm.pickle"
    prm_path = "output/max_nodes-fhw1000-obs-thres0.45-k_nearest-7-connection_radius-5.0-date-2020-10-11-02-53-23/prm.pickle"

    gng_top_path = "../persistence/output/exp_factor-fhw9-is-topology-feedback-False-is-bias-sampling-True-bias_ratio-0.75-max_epoch-400-max_edge_age-56-date-2020-10-11-03-53-40/gng400.pickle"
    gng_path = "../persistence/output/exp_factor-fhw9-is-topology-feedback-False-is-bias-sampling-False-bias_ratio-0.75-max_epoch-400-max_edge_age-56-date-2020-10-11-03-36-54/gng400.pickle"
    gng_top_feedback_path = "../persistence/output/exp_factor-fhw9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.75-max_epoch-400-max_edge_age-56-date-2020-10-11-03-16-23/gng400.pickle"
    """
    prm_dense_path = "output/max_nodes-intel4000-obs-thres0.25-k_nearest-7-connection_radius-5.0-date-2020-10-11-15-03-54/prm.pickle"
    prm_dense_hilbert_path = "output/max_nodes-intel2000-obs-thres0.25-k_nearest-7-connection_radius-5.0-date-2020-10-11-15-02-45/prm.pickle"
    prm_path = "output/max_nodes-intel1000-obs-thres0.25-k_nearest-7-connection_radius-5.0-date-2020-10-11-15-00-52/prm.pickle"

    gng_top_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-False-is-bias-sampling-True-bias_ratio-0.75-max_epoch-500-max_edge_age-56-date-2020-10-11-17-08-42/gng300.pickle"
    gng_top_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-False-is-bias-sampling-True-bias_ratio-0.75-max_epoch-300-max_edge_age-70-date-2020-10-11-18-05-56/gng200.pickle"
    gng_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-False-is-bias-sampling-False-bias_ratio-0.75-max_epoch-500-max_edge_age-56-date-2020-10-11-15-46-32/gng300.pickle"
    gng_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-False-is-bias-sampling-False-bias_ratio-0.75-max_epoch-300-max_edge_age-70-date-2020-10-11-18-01-43/gng200.pickle"
    gng_top_feedback_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.75-max_epoch-500-max_edge_age-56-date-2020-10-11-15-27-01/gng400.pickle"
    gng_top_feedback_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.75-max_epoch-500-max_edge_age-56-date-2020-10-11-17-17-26/gng150.pickle"
    gng_top_feedback_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.78-max_epoch-300-max_edge_age-56-date-2020-10-11-17-30-49/gng200.pickle"
    gng_top_feedback_path = "../persistence/output/exp_factor-intel9-is-topology-feedback-True-is-bias-sampling-True-bias_ratio-0.75-max_epoch-300-max_edge_age-70-date-2020-10-11-17-53-22/gng200.pickle"
    with open("test_samples/intel_crap.pickle", 'rb') as tf:
        test_data = pickle.load(tf)
    goal_list = test_data[0]
    start_list = test_data[1]

    for roadmap in roadmap_types:
        if roadmap == "gng":
            prm_graph = convert_gng_to_nxgng(load_graph(gng_path), map_array, obstacle_threshold, resolution)
        elif roadmap == "gng_top":
            prm_graph = convert_gng_to_nxgng(load_graph(gng_top_path), map_array, obstacle_threshold, resolution)
        elif roadmap == "gng_top_feedback":
            prm_graph = convert_gng_to_nxgng(load_graph(gng_top_feedback_path), map_array, obstacle_threshold, resolution)
        elif roadmap == "prm":
            prm_graph = load_graph(prm_path)
            prm_graph = filter_graph_with_ground_truth_map(prm_graph, map_array, obstacle_threshold, resolution)
        elif roadmap == "prm_dense_hilbert":
            prm_graph = load_graph(prm_dense_hilbert_path)
            prm_graph = filter_graph_with_ground_truth_map(prm_graph, map_array, obstacle_threshold, resolution)
        else:  # roadmap == "prm_dense":
            prm_graph = load_graph(prm_dense_path)
            prm_graph = filter_graph_with_ground_truth_map(prm_graph, map_array, obstacle_threshold, resolution)

        path_sets = []
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
            if roadmap == "gng" or roadmap == "gng_top" or roadmap == "gng_top_feedback" or roadmap == "prm" or roadmap == "prm_dense_hilbert" or roadmap == "prm_dense":
                full_graph = add_start_n_goal_to_graph(prm_graph.copy(), start_loc, goal_loc, save_pickle, map_array)
            #else:
                #full_graph = add_start_n_goal_to_graph(prm_graph.copy(), start_loc, goal_loc, save_pickle, ground_map_array)
            #full_graph = filter_graph_with_ground_truth_map(full_graph, ground_map_array, obstacle_threshold, ground_resolution)

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
            path_sets.append(path_nodes)
            if not save_pickle:
                if roadmap == "prm_dense":
                    save_img(ground_map_data, full_graph, start_loc, start_node, goal_loc, goal_node,
                             path_nodes, data_save_dic[roadmap], fig_num=lamda_, save_graph=False, figure_cmap="viridis")
                else:
                    save_img(map_data, full_graph, start_loc, start_node, goal_loc, goal_node,
                           path_nodes, data_save_dic[roadmap], fig_num=lamda_, save_graph=False)
        #plot_all_paths(map_data, path_sets, full_graph)
        print("############## for ", roadmap)
        print("success trial", np.sum(success_list))
        print("success_list:", success_list)
        print("nodes explored list",node_explored_list)
        print("distance to goal list", distance_to_goal_list)
        if save_pickle:
            if roadmap == "gng":
                with open(data_save_dic[roadmap] + 'freiburg_gng1208_200.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "gng_top":
                with open(data_save_dic[roadmap] + 'freiburg_gngtop_1208_200.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "gng_top_feedback":
                with open(data_save_dic[roadmap] + 'freiburg_gngtop_feedback_1208_200.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "prm":
                with open(data_save_dic[roadmap] + 'freiburg_prm1208.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "prm_dense":
                with open(data_save_dic[roadmap] + 'freiburg_prmdense_2500.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)
            elif roadmap == "prm_dense_hilbert":
                with open(data_save_dic[roadmap] + 'freiburg_prmdense_hilbert4000.pickle', 'wb') as handle:
                    pickle.dump([success_list,node_explored_list, distance_to_goal_list], handle)

