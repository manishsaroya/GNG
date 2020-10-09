"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import pickle
import matplotlib.pyplot as plt
from persistence.utils import load_hilbert_map, convert_map_dict_to_array, collision_check, convert_gng_to_nxgng, get_topological_accuracy, load_graph
from persistence.drive_hilbert_persistence import get_top_n_persistence_node_location
import networkx as nx
import numpy as np
from bresenham import bresenham
from bfs_looping import BFSLoopDetection, get_directed_intersection_attribute, get_polygon_loop
import math

# def collision_check(map_array, pos1, pos2, obstacle_threshold, resolution):
# 	"""
# 	Collision checker function between pos1 and pose2 via bresenham pixel selection
# 	:param map_array: map data in array form
# 	:param pos1: position of point1
# 	:param pos2: position of point2
# 	:param obstacle_threshold: obstacle threshold
# 	:return: Bool value true if clear path exist
# 	"""
# 	# convert pos1 and pose2 in indices
# 	ipos1 = [int(pos1[0] * (1 / resolution) + 160), int(pos1[1] * (1 / resolution) + 160)]
# 	ipos2 = [int(pos2[0] * (1 / resolution) + 160), int(pos2[1] * (1 / resolution) + 160)]
# 	check_list = list(bresenham(ipos1[0], ipos1[1], ipos2[0], ipos2[1]))
# 	for cell in check_list:
# 		if map_array[cell[0]][cell[1]] > obstacle_threshold and (not np.isnan(map_array[cell[0]][cell[1]])):
# 			return False
# 	return True

# def convert_gng_to_nxgng(path, map_array, obs_threshold, resolution):
# 	# Also removes edges based on hilbert threshold
#     with open(path, 'rb') as tf:
#         g = pickle.load(tf)
#     nxgraph = nx.Graph()
#     nodeid = {}
#     for indx, node in enumerate(g.graph.nodes):
#         nodeid[node] = indx
#         nxgraph.add_node(nodeid[node], pos=(node.weight[0][0], node.weight[0][1]))
#     positions = nx.get_node_attributes(nxgraph, "pos")
#     for node_1, node_2 in g.graph.edges:
#         if collision_check(map_array, positions[nodeid[node_1]],
#                                            positions[nodeid[node_2]], obs_threshold, resolution):
#             nxgraph.add_edge(nodeid[node_1], nodeid[node_2])
#     return nxgraph


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
			plt.setp(line, linewidth=1, color='lightsteelblue')

	# plot path
	if path_nodes is not None:
		position = nx.get_node_attributes(graph, 'pos')
		for path_n in path_nodes:
			plt.scatter(position[path_n][0], position[path_n][1], s=40, marker='*', facecolors='red')
	if samples is not None:
		plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1], marker='v', facecolors='cyan')

	if persistence_birthnode is not None:
		for i in range(len(persistence_birthnode)):
			plt.plot(persistence_birthnode[i][0], persistence_birthnode[i][1], "b*", markersize=25)

	if persistence_1homnode is not None:
		for i in range(len(persistence_1homnode)):
			plt.plot(persistence_1homnode[i][0], persistence_1homnode[i][1], "y*", markersize=10)

	if show:
		plt.savefig(dir + "graph{}.eps".format(fignum))

# def get_topological_accuracy(gng_, feature):
# 	topological_accuracy = []
# 	position = nx.get_node_attributes(gng_, 'pos')
# 	local_distance = 5
# 	######### write code for inside detection
# 	for f_indx, f in enumerate(feature):
# 		##### find the nearest vertex
# 		short_distance = 10000
# 		closest_node = None
# 		local_graph = nx.Graph()
# 		for indx, node in enumerate(gng_.nodes):
# 			pose = position[node]
# 			distance = math.sqrt((f[0] - pose[0]) ** 2 + (f[1] - pose[1]) ** 2)
# 			if short_distance > distance:  # TODO: Check if the closest node not in the local_graph
# 				closest_node = indx
# 				short_distance = distance
# 			if local_distance > distance:
# 				local_graph.add_node(indx, pos=pose)
#
# 		for node1, node2 in gng_.edges:
# 			if (node1 in local_graph.nodes) and (node2 in local_graph.nodes):
# 				local_graph.add_edge(node1, node2)
#
# 		loopfinder = BFSLoopDetection()
# 		loops = loopfinder.get_breadth_first_search_loops(local_graph, closest_node)
# 		#print(loops)
#
# 		collisions = get_directed_intersection_attribute(local_graph, f)
# 		is_inside, polygon = get_polygon_loop(loops, collisions)
# 		#print(f_indx, is_inside, polygon)
# 		topological_accuracy.append(is_inside)
# 		#draw_image(data, resolution, "", f_indx, persistence_birthnode=None, graph=local_graph, path_nodes=polygon, persistence_1homnode=[f], show=True)
# 	return topological_accuracy

if __name__ == "__main__":
	path = "output/exp_factor-freiburg20-is-bias-sampling-True-bias_ratio-0.75-max_epoch-400-max_edge_age-40-date-2020-10-04-23-01-23/gng100.pickle"
	feature = [[-14.000000000000057, 2.799999999999955]]
	data, resolution = load_hilbert_map(map_type="freiburg")
	map_array = convert_map_dict_to_array(data, resolution)
	gng_ = convert_gng_to_nxgng(load_graph(path), map_array, 0.6, resolution)
	feature, persistence_1hom_weights = get_top_n_persistence_node_location(25, "freiburg",
																	  location_type="death", feature_type=1)
	# with open(path, 'rb') as tf:
	# 	g = pickle.load(tf)

	# create the graph based on the feature nearby area

	accuracy = get_topological_accuracy(gng_, feature, 5)
	print(feature)
	print(accuracy)
	accuracy_indices = [i for i, val in enumerate(accuracy) if val]
	print(accuracy_indices)
	for i in sorted(accuracy_indices, reverse=True):
		del feature[i]
	print(feature)