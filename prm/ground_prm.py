"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import networkx as nx
import numpy as np
from persistence.utils import load_hilbert_map, convert_map_dict_to_array
import matplotlib.pyplot as pl
from sklearn.neighbors import NearestNeighbors
from bresenham import bresenham
import os
from math import sqrt
import argparse
import datetime
import pickle
from scipy.spatial import KDTree
from prm import hilbert_samples, save_img


class CollisionChecker:
	def __init__(self, map_data, resolution=3):
		self.map_data = map_data
		self.tree = KDTree(self.map_data["Xq"])
		self.resolution = resolution



	def is_collision(self, start, end):
		distance = sqrt((start[0] - end[0])**2 + (start[1] - end[1])**2)
		num_points = int(distance * 2 * self.resolution) + 1

		equidistant_points = list(zip(np.linspace(start[0], end[0], num_points),
								 np.linspace(start[1], end[1], num_points)))

		dist, self.indices = self.tree.query(equidistant_points, distance_upper_bound=2 / (self.resolution))
		if np.isinf(dist).any():
			return True
		else:
			try:
				intermediate_points = [map_data["Xq"][i] for i in self.indices]
			except IndexError:
				pl.scatter(self.map_data['Xq'][:, 0], self.map_data['Xq'][:, 1], c=self.map_data['yq'], cmap='jet', s=70, vmin=0, vmax=1,
						   edgecolors='')
				pl.colorbar()
				pl.scatter(list(zip(*equidistant_points))[0], list(zip(*equidistant_points))[1], c="cyan")
				pl.show()
				pass
			#print(intermediate_points)
			obs = [map_data["yq"][i] for i in self.indices]
			return sum(obs)!=0
			#grid_values = df_points.groupby(indices).v.sum()



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--number_of_samples', type=int, default=10000)
	parser.add_argument('--exp_factor', type=int, default=30)
	parser.add_argument('--obstacle_threshold', type=float, default=0.4)
	parser.add_argument('--max_nodes', type=int, default=1500)
	parser.add_argument('--k_nearest', type=int, default=9)
	parser.add_argument('--log_dir', type=str, default='./output')
	parser.add_argument('--connection_radius', type=float, default=5.0)
	args = parser.parse_args()
	args.log_dir = './output/max_nodes-' + str(args.max_nodes) + "-k_nearest-" + \
				   str(args.k_nearest) + "-connection_radius-" + str(args.connection_radius) + "-date-" + \
				   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '/'

	# if not os.path.exists(args.log_dir):
	#	os.makedirs(args.log_dir)
	with open("ground_map.pickle", 'rb') as tf:
		df_grid = pickle.load(tf)

	map_data = {"Xq": np.array(list(zip(list(df_grid.x), list(df_grid.y)))), "yq": np.array(list(df_grid.v))}
	resolution = 3
	colcheck = CollisionChecker(map_data, resolution)
	#is_col = colcheck.is_collision([-5,-20], [-7,-20])
	# load map
	# map_data, resolution = load_hilbert_map(map_type="intel")
	#map_array = convert_map_dict_to_array(map_data, resolution)
	# get samples from hilbert maps
	sample_list = hilbert_samples(map_data.copy(), args.exp_factor, num_samples=args.number_of_samples)
	#sample_list = get_ground_map(ground_path)
	# take unique samples
	sample_list = [list(t) for t in set(tuple(element) for element in sample_list)]
	# truncated based on max nodes
	sample_list = sample_list[:args.max_nodes]
	# find k nearest neighbor
	nbrs = NearestNeighbors(n_neighbors=args.k_nearest, algorithm='ball_tree').fit(sample_list)
	distances, indices = nbrs.kneighbors(sample_list)
	pl.scatter(map_data['Xq'][:, 0], map_data['Xq'][:, 1], c=map_data['yq'], cmap='jet', s=70, vmin=0,
			   vmax=1,
			   edgecolors='')
	pl.colorbar()
	pl.scatter(list(zip(*sample_list))[0], list(zip(*sample_list))[1], c="cyan", s=10)
	pl.show()
	# create gragh
	prm_graph = nx.Graph()
	# add graph nodes
	for indx, s in enumerate(sample_list):
		prm_graph.add_node(indx, pos=(s[0], s[1]))
	# add graph edges
	for row, node_adjacency_list in enumerate(indices):
		for column, other_node in enumerate(node_adjacency_list):
			distance_metric = distances[row][column] < args.connection_radius
			collision_metric = colcheck.is_collision(sample_list[node_adjacency_list[0]],
													 sample_list[other_node])
			#collision_metric = collision_check(map_array, sample_list[node_adjacency_list[0]],
			#								   sample_list[other_node], args.obstacle_threshold, resolution)
			validation_metric = node_adjacency_list[0] != other_node
			if distance_metric and collision_metric and validation_metric:
				prm_graph.add_edge(node_adjacency_list[0], other_node, distance=distances[row][column])
	print("rendering")
	save_img(map_data, prm_graph, "")
