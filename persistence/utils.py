"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu

Description: Contains support functions to create persistence for GNG graph and hilbert map
"""
import pickle
import numpy as np
import networkx as nx
from gudhi import SimplexTree
import pdb
def load_hilbert_map(map_type='intel'):
    """
    :param map_type: type of map to load, eg "intel" or "drive"
    :return: a dictionary with key 'Xq' as locations and key 'yq' as occupancy probabilities
    """
    mapdata = {}
    my_map = map_type
    if my_map == "drive":
        with open('./../dataset/mapdata_{}.pickle'.format(0), 'rb') as tf:
            mapdata_ = pickle.load(tf)
            mapdata['Xq'] = mapdata_['Xq'].numpy()
            mapdata['yq'] = mapdata_['yq'].numpy()
        for i in range(1, 4):
            with open('./../dataset/mapdata_{}.pickle'.format(i), 'rb') as tf:
                mapdata_ = pickle.load(tf)
                mapdata['Xq'] = np.concatenate((mapdata.get('Xq'), mapdata_['Xq'].numpy()), axis=0)
                mapdata['yq'] = np.concatenate((mapdata.get('yq'), mapdata_['yq'].numpy()), axis=0)
    else:
        with open('./../dataset/mapdata_{}.pickle'.format(271), 'rb') as tf:
            mapdata = pickle.load(tf)
        # convert to numpy
        mapdata['Xq'] = mapdata['X']
        mapdata['yq'] = mapdata['Y']
        #pdb.set_trace()
    return mapdata


def load_graph(path):
    """
    :param path: path to the graph to be loaded
    :return: graph
    """
    with open(path, 'rb') as tf:
        g = pickle.load(tf)
    return g


def convert_map_dict_to_array(map_dict):
    """
    :param map_dict: a dictionary with key 'Xq' as locations and key 'yq' as occupancy probabilities
    :return: hilbert map in array form
    Note: Hard code for resolution 0.5 and map (-80,80)
    """
    resolution = 0.3
    map_array = np.zeros([320, 320]) + 0.5
    for indx, point in enumerate(map_dict['Xq']):
        map_array[int(point[0] * (1/resolution) + 160)][int(point[1] * (1/resolution) + 160)] = map_dict['yq'][indx]
    return map_array


def create_intensity_graph(_graph, graph_type):
    """
    :param _graph: networkx graph
    :return: Add intensity value to every node from the hilbert_map and return the graph
    """
    map_dict = load_hilbert_map(graph_type)
    map_array = convert_map_dict_to_array(map_dict)

    # assign intensities to graph based on the map array
    position = nx.get_node_attributes(_graph, "pos")
    # make attribute dictionary
    intensity = {}
    resolution = 0.3
    for i in range(_graph.number_of_nodes()):
        intensity[i] = map_array[int(position[i][0]*(1/resolution)) + 160, int(position[i][1]*(1/resolution)) + 160]  # approx interpolation
    nx.set_node_attributes(_graph, intensity, "intensity")
    return _graph


def create_simplex_from_graph(G):
    st = SimplexTree()
    node_values = nx.get_node_attributes(G, "intensity")
    print("node intensities", node_values)
    for clique in nx.enumerate_all_cliques(G):
        clique_value = node_values[clique[0]]
        for n in clique:
            # take max values
            if clique_value < node_values[n]:
                clique_value = node_values[n]
        st.insert(clique, clique_value)
    return st


def print_complex_attributes(cmplx):
    result_str = 'num_vertices=' + repr(cmplx.num_vertices())
    print(result_str)
    result_str = 'num_simplices=' + repr(cmplx.num_simplices())
    print(result_str)

    if len(cmplx.get_skeleton(2)) < 20:
        print("skeleton(2) =")
        for sk_value in cmplx.get_skeleton(2):
            print(sk_value)