"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""

import matplotlib.pyplot as plt
from persistence.utils import load_hilbert_map, convert_map_dict_to_array, convert_gng_to_nxgng, load_graph, get_0hom_topological_accuracy
from persistence.drive_hilbert_persistence import get_top_n_persistence_node_location
import networkx as nx
import numpy as np
from bfs_looping import BFSLoopDetection, get_directed_intersection_attribute, get_polygon_loop
import math



def draw_image(data,resolution, dir, fignum, graph=None, persistence_birthnode=None, persistence_1homnode=None, samples=None, show=True, path_nodes=None):
	fig = plt.figure(figsize=(40/4, 35/4))
	plt.axis("equal")
	plt.style.use('seaborn-dark')
	#plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap="jet", s=(70/0.3) * resolution*0.2, vmin=0, vmax=1, edgecolors='')
	plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], s=10, vmin=0, vmax=1, edgecolors='')
	plt.colorbar(fraction= 0.047, pad=0.02)
	if graph is not None:
		position = nx.get_node_attributes(graph, 'pos')
		# Plot the graph
		for node_1, node_2 in graph.edges:
			weights = np.concatenate([[position[node_1]], [position[node_2]]])
			line, = plt.plot(*weights.T, color='lightsteelblue')
			plt.setp(line, linewidth=1.5, color='lightsteelblue')

	# plot path
	if path_nodes is not None:
		position = nx.get_node_attributes(graph, 'pos')
		for path_n in path_nodes:
			plt.scatter(position[path_n][0], position[path_n][1], s=40, marker='*', facecolors='red')
	if samples is not None:
		plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1], marker='v', facecolors='cyan')

	if persistence_birthnode is not None:
		for i in range(len(persistence_birthnode)):
			plt.plot(persistence_birthnode[i][0], persistence_birthnode[i][1], "*", markersize=10, markerfacecolor="None", markeredgecolor="blue", markeredgewidth=0.2)

	if persistence_1homnode is not None:
		for i in range(len(persistence_1homnode)):
			plt.plot(persistence_1homnode[i][0], persistence_1homnode[i][1], "y*", markersize=10)

	if show:
		plt.savefig(dir + "graph0hom{}.eps".format(fignum))


# def count_components(g):
# 	is_connected = False
# 	explored = {}
# 	for node in g.nodes:
# 		explored[node] = False
# 	num_connected_components = 0
# 	while not all(value==True for value in explored.values()):
# 		for key, value in explored.items():
# 			if not value:
# 				start = key
# 				break
# 		queue = [start]
# 		explored[start] = True
# 		node_explored_count = 0
# 		while len(queue) != 0:
# 			node = queue.pop(0)
# 			for adj in g.neighbors(node):
# 				if not explored[adj]:
# 					explored[adj] = True
# 					queue.append(adj)
# 			node_explored_count += 1
# 		if node_explored_count > 1:
# 			num_connected_components += 1
# 		print("component=", num_connected_components, "num_nodes=", node_explored_count)
# 	if num_connected_components == 1:
# 		is_connected = True
# 	return is_connected, num_connected_components
#
#
#
#
# def get_0hom_topological_accuracy(gng_, feature, local_distance):
# 	topological_accuracy_0hom = []
# 	position = nx.get_node_attributes(gng_, 'pos')
#
# 	######### write code for inside detection
# 	for f_indx, f in enumerate(feature):
# 		local_graph = nx.Graph()
# 		for indx, node in enumerate(gng_.nodes):
# 			pose = position[node]
# 			distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
# 			if local_distance > distance:
# 				local_graph.add_node(indx, pos=pose)
#
# 		for node1, node2 in gng_.edges:
# 			if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
# 				local_graph.add_edge(node1, node2)
#
# 		is_connected, num_components = count_components(local_graph)
#
# 		print(f_indx, is_connected, num_components)
# 		topological_accuracy_0hom.append(is_connected)
# 	return topological_accuracy_0hom

if __name__ == "__main__":
	path = "output/exp_factor-freiburg20-is-bias-sampling-True-bias_ratio-0.75-max_epoch-400-max_edge_age-50-date-2020-10-07-13-12-23/gng300.pickle"
	data, resolution = load_hilbert_map(map_type="freiburg")
	map_array = convert_map_dict_to_array(data, resolution)
	gng_ = convert_gng_to_nxgng(load_graph(path), map_array, 0.6, resolution)
	feature, persistence_1hom_weights = get_top_n_persistence_node_location(5, "freiburg",
																	  location_type="death", feature_type=0)
	# with open(path, 'rb') as tf:
	# 	g = pickle.load(tf)

	# create the graph based on the feature nearby area
	# topological_accuracy_0hom = []
	# position = nx.get_node_attributes(gng_, 'pos')
	# local_distance = 1.1
	# ######### write code for inside detection
	# for f_indx, f in enumerate(feature):
	# 	local_graph = nx.Graph()
	# 	for indx, node in enumerate(gng_.nodes):
	# 		pose = position[node]
	# 		distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
	# 		if local_distance > distance:
	# 			local_graph.add_node(indx, pos=pose)
	#
	# 	for node1, node2 in gng_.edges:
	# 		if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
	# 			local_graph.add_edge(node1, node2)
	#
	# 	is_connected, num_components = count_components(local_graph)
	#
	# 	print(f_indx, is_connected, num_components)
	# 	topological_accuracy_0hom.append(is_connected)
	# 	draw_image(data, resolution, "", f_indx, persistence_birthnode=[f], graph=local_graph, path_nodes=None, persistence_1homnode=None, show=True)
	accuracy = get_0hom_topological_accuracy(gng_, feature, 1.1) #topological_accuracy_0hom #get_topological_accuracy(gng_, feature, 1)

	print(feature)
	print(accuracy)
	accuracy_indices = [i for i, val in enumerate(accuracy) if val]
	print(accuracy_indices)
	for i in sorted(accuracy_indices, reverse=True):
		del feature[i]
	print(feature)