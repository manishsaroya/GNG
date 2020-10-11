"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import networkx as nx
import numpy as np
from persistence.utils import load_hilbert_map, convert_map_dict_to_array, collision_check
import matplotlib.pyplot as pl
from sklearn.neighbors import NearestNeighbors
from bresenham import bresenham
import os
import argparse
import datetime
import pickle


def save_img(data, graph, dir, save_data=True, save_graph=True):
	"""
	:param data: map data dictionary
	:param graph: prm nx graph
	:param dir: output directory path
	:return: save prm graph and visualization image to output directory
	"""
	fig = pl.figure(figsize=(40/4, 35/4))
	ax = fig.add_subplot(111)
	pl.axis("equal")
	pl.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap='viridis', s=70, vmin=0, vmax=1, edgecolors='')
	pl.colorbar()
	position = nx.get_node_attributes(graph, 'pos')
	for node_1, node_2 in graph.edges:
		weights = np.concatenate([[position[node_1]], [position[node_2]]])
		line, = pl.plot(*weights.T, color='lightsteelblue')
		pl.setp(line, linewidth=2, color='lightsteelblue')
	#pl.title('Test image')
	if save_data:
		pl.savefig(dir + "prm.png")
	if save_graph:
		with open(dir + 'prm.pickle', 'wb') as handle:
			pickle.dump(graph, handle)


def hilbert_samples(map_data, exp_factor, num_samples=600):
	"""
	:param map_data: map info
	:param exp_factor: exponential factor to sample more from free space
	:param num_samples: number of samples required
	:return: samples
	"""
	map_data['yq'] = np.ones(len(map_data['yq'])) - map_data['yq']
	#map_data['yq'] = np.exp(exp_factor * map_data['yq'])
	map_data["yq"][map_data["yq"] < 0.5] = 0
	# normalize the probabilities
	map_data['yq'] /= np.linalg.norm(map_data['yq'], ord=1)

	samples_list = map_data['Xq'][np.random.choice(len(map_data['Xq']), size=num_samples, p=map_data['yq'])]
	return samples_list


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--number_of_samples', type=int, default=15000)
	parser.add_argument('--exp_factor', type=int, default=20)
	parser.add_argument('--obstacle_threshold', type=float, default=0.5)
	parser.add_argument('--max_nodes', type=int, default=2000)
	parser.add_argument('--k_nearest', type=int, default=7)
	parser.add_argument('--log_dir', type=str, default='./output')
	parser.add_argument('--connection_radius', type=float, default=5.0)
	parser.add_argument('--map_type', type=str, default="freiburg")
	args = parser.parse_args()
	args.log_dir = './output/max_nodes-' + args.map_type + str(args.max_nodes) + "-obs-thres" + str(args.obstacle_threshold) +\
				   "-k_nearest-" + \
				   str(args.k_nearest) + "-connection_radius-" + str(args.connection_radius) + "-date-" + \
				   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '/'

	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	# load map
	map_data, resolution = load_hilbert_map(map_type=args.map_type)
	#resolution = 0.2
	#with open("freiburg_ground_map_q_resolution_final.pickle", 'rb') as tf:
	#	map_data = pickle.load(tf)
	map_array = convert_map_dict_to_array(map_data, resolution)
	map_data["yq"] = 1.0 * (map_data["yq"] > args.obstacle_threshold)
	# get samples from hilbert maps
	sample_list = hilbert_samples(map_data.copy(), args.exp_factor, num_samples=args.number_of_samples)
	# take unique samples
	sample_list = [list(t) for t in set(tuple(element) for element in sample_list)]
	# truncated based on max nodes
	sample_list = sample_list[:args.max_nodes]
	# find k nearest neighbor
	nbrs = NearestNeighbors(n_neighbors=args.k_nearest, algorithm='ball_tree').fit(sample_list)
	distances, indices = nbrs.kneighbors(sample_list)
	# create gragh
	prm_graph = nx.Graph()
	# add graph nodes
	for indx, s in enumerate(sample_list):
		prm_graph.add_node(indx, pos=(s[0], s[1]))
	# add graph edges
	for row, node_adjacency_list in enumerate(indices):
		for column, other_node in enumerate(node_adjacency_list):
			distance_metric = distances[row][column] < args.connection_radius
			collision_metric = collision_check(map_array, sample_list[node_adjacency_list[0]],
											   sample_list[other_node], args.obstacle_threshold, resolution)
			validation_metric = node_adjacency_list[0] != other_node
			if distance_metric and collision_metric and validation_metric:
				prm_graph.add_edge(node_adjacency_list[0], other_node, distance=distances[row][column])
	print("rendering")
	save_img(map_data, prm_graph, args.log_dir)
