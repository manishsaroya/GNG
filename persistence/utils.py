"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu

Description: Contains support functions to create persistence for GNG graph and hilbert map
"""
import pickle
import numpy as np
import networkx as nx
from gudhi import SimplexTree
from persistence.bfs_looping import BFSLoopDetection, get_directed_intersection_attribute, get_polygon_loop
from bresenham import bresenham
import math


def load_hilbert_map(map_type='intel'):
	"""
    :param map_type: type of map to load, eg "intel" or "drive"
    :return: a dictionary with key 'Xq' as locations and key 'yq' as occupancy probabilities
    """
	mapdata = {}
	my_map = map_type
	resolution = None
	if my_map == "drive":
		resolution = 0.5
		with open('./../dataset/mapdata_{}.pickle'.format(0), 'rb') as tf:
			mapdata_ = pickle.load(tf)
			mapdata['Xq'] = mapdata_['Xq'].numpy()
			mapdata['yq'] = mapdata_['yq'].numpy()
		for i in range(1, 4):
			with open('./../dataset/mapdata_{}.pickle'.format(i), 'rb') as tf:
				mapdata_ = pickle.load(tf)
				mapdata['Xq'] = np.concatenate((mapdata.get('Xq'), mapdata_['Xq'].numpy()), axis=0)
				mapdata['yq'] = np.concatenate((mapdata.get('yq'), mapdata_['yq'].numpy()), axis=0)
	elif my_map == "freiburg":
		resolution = 0.2
		with open('./../dataset/mapdata_{}.pickle'.format(4798), 'rb') as tf:
			# with open('./dataset/mapdata_{}.pickle'.format(908), 'rb') as tf:
			mapdata = pickle.load(tf)
		# convert to numpy
		mapdata['Xq'] = mapdata['X']
		mapdata['yq'] = mapdata['Y']
		# pdb.set_trace()
	elif my_map == "fhw":
		resolution = 0.2
		with open('./../dataset/mapdata_{}.pickle'.format(499), 'rb') as tf:
			# with open('./dataset/mapdata_{}.pickle'.format(908), 'rb') as tf:
			mapdata = pickle.load(tf)
		# convert to numpy
		mapdata['Xq'] = mapdata['X']
		mapdata['yq'] = mapdata['Y']
		# pdb.set_trace()
	else:
		resolution = 0.3
		with open('./../dataset/mapdata_{}.pickle'.format(908), 'rb') as tf:
			# with open('./dataset/mapdata_{}.pickle'.format(908), 'rb') as tf:
			mapdata = pickle.load(tf)
		# convert to numpy
		mapdata['Xq'] = mapdata['X']
		mapdata['yq'] = mapdata['Y']
		# pdb.set_trace()
	return mapdata, resolution


def load_graph(path):
	"""
    :param path: path to the graph to be loaded
    :return: graph
    """
	with open(path, 'rb') as tf:
		g = pickle.load(tf)
	return g


def convert_map_dict_to_array(map_dict, resolution):
	"""
    :param map_dict: a dictionary with key 'Xq' as locations and key 'yq' as occupancy probabilities
    :return: hilbert map in array form
    Note: Hard code for resolution 0.5 and map (-80,80)
    """
	# resolution = 0.3
	map_array = np.zeros([600, 600]) + 0.5
	for indx, point in enumerate(map_dict['Xq']):
		map_array[int(point[0] * (1 / resolution) + 300)][int(point[1] * (1 / resolution) + 300)] = map_dict['yq'][indx]
	return map_array


def collision_check(map_array, pos1, pos2, obstacle_threshold, resolution):
	"""
	Collision checker function between pos1 and pose2 via bresenham pixel selection
	:param map_array: map data in array form
	:param pos1: position of point1
	:param pos2: position of point2
	:param obstacle_threshold: obstacle threshold
	:return: Bool value true if clear path exist
	"""
	# convert pos1 and pose2 in indices
	ipos1 = [int(pos1[0] * (1 / resolution) + 300), int(pos1[1] * (1 / resolution) + 300)]
	ipos2 = [int(pos2[0] * (1 / resolution) + 300), int(pos2[1] * (1 / resolution) + 300)]
	check_list = list(bresenham(ipos1[0], ipos1[1], ipos2[0], ipos2[1]))
	for cell in check_list:
		if map_array[cell[0]][cell[1]] > obstacle_threshold and (not np.isnan(map_array[cell[0]][cell[1]])):
			return False
	return True

def create_intensity_graph(_graph, graph_type):
	"""
    :param _graph: networkx graph
    :return: Add intensity value to every node from the hilbert_map and return the graph
    """
	map_dict, resolution = load_hilbert_map(graph_type)
	map_array = convert_map_dict_to_array(map_dict, resolution)

	# assign intensities to graph based on the map array
	position = nx.get_node_attributes(_graph, "pos")
	# make attribute dictionary
	intensity = {}
	# resolution = 0.3
	for i in range(_graph.number_of_nodes()):
		intensity[i] = map_array[int(position[i][0] * (1 / resolution)) + 300, int(
			position[i][1] * (1 / resolution)) + 300]  # approx interpolation
	nx.set_node_attributes(_graph, intensity, "intensity")
	return _graph, map_dict


def convert_gng_to_nxgng(g, map_array, obs_threshold, resolution):
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


def get_topological_accuracy(gng_, feature, local_distance):
	topological_accuracy = []
	position = nx.get_node_attributes(gng_, 'pos')
	# write code for inside detection
	for f_indx, f in enumerate(feature):
		# find the nearest vertex
		short_distance = 10000
		closest_node = None
		local_graph = nx.Graph()
		for indx, node in enumerate(gng_.nodes):
			pose = position[node]
			distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
			if short_distance > distance:  # TODO: Check if the closest node not in the local_graph
				closest_node = indx
				short_distance = distance
			if local_distance > distance:
				local_graph.add_node(indx, pos=pose)

		for node1, node2 in gng_.edges:
			if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
				local_graph.add_edge(node1, node2)

		loopfinder = BFSLoopDetection()
		loops = loopfinder.get_breadth_first_search_loops(local_graph, closest_node)
		# print(loops)

		collisions = get_directed_intersection_attribute(local_graph, f)
		is_inside, polygon = get_polygon_loop(loops, collisions)
		# print(f_indx, is_inside, polygon)
		topological_accuracy.append(is_inside)
	# draw_image(data, resolution, "", f_indx, persistence_birthnode=None, graph=local_graph,
	# path_nodes=polygon, persistence_1homnode=[f], show=True)
	return topological_accuracy


def count_components(g):
	is_connected = False
	explored = {}
	for node in g.nodes:
		explored[node] = False
	num_connected_components = 0
	while not all(value==True for value in explored.values()):
		for key, value in explored.items():
			if not value:
				start = key
				break
		queue = [start]
		explored[start] = True
		node_explored_count = 0
		while len(queue) != 0:
			node = queue.pop(0)
			for adj in g.neighbors(node):
				if not explored[adj]:
					explored[adj] = True
					queue.append(adj)
			node_explored_count += 1
		if node_explored_count > 1:
			num_connected_components += 1
		#print("component=", num_connected_components, "num_nodes=", node_explored_count)
	if num_connected_components == 1:
		is_connected = True
	return is_connected, num_connected_components

def get_0hom_topological_accuracy(gng_, feature, local_distance):
	topological_accuracy_0hom = []
	position = nx.get_node_attributes(gng_, 'pos')

	for f_indx, f in enumerate(feature):
		local_graph = nx.Graph()
		for indx, node in enumerate(gng_.nodes):
			pose = position[node]
			distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
			if local_distance > distance:
				local_graph.add_node(indx, pos=pose)

		for node1, node2 in gng_.edges:
			if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
				local_graph.add_edge(node1, node2)

		is_connected, num_components = count_components(local_graph)

		#print(f_indx, is_connected, num_components)
		topological_accuracy_0hom.append(is_connected)
	return topological_accuracy_0hom
