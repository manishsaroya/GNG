# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""Initialize module utils."""

import numpy as np
import networkx as nx
import imageio
from matplotlib import pylab as pl
import os
import glob
from past.builtins import xrange
from future.utils import iteritems
import pickle
import copy
import re
import numpy
import sys
pos = None
G = None
map_image = None
numpy.set_printoptions(threshold=sys.maxsize)
import pdb

def read_file_hilbert_maps(map_ ='intel'):
	"""Create the graph and returns the networkx version of it 'G'."""
	global pos
	global G
	global map_image

	if map_ == 'intel':
		with open('./dataset/mapdata_{}.pickle'.format(271),'rb') as tf:
			mapdata = pickle.load(tf)
		# convert to numpy 
		mapdata['Xq'] = mapdata['X']
		mapdata['yq'] = mapdata['Y']
	else:
		mapdata = {}
		with open('./dataset/mapdata_{}.pickle'.format(0),'rb') as tf:
			mapdata_ = pickle.load(tf)
			mapdata['Xq'] = mapdata_['Xq'].numpy()
			mapdata['yq'] = mapdata_['yq'].numpy()
		for i in range(1,4):
			with open('./dataset/mapdata_{}.pickle'.format(i),'rb') as tf:
				mapdata_ = pickle.load(tf)
				mapdata['Xq'] = np.concatenate((mapdata.get('Xq'), mapdata_['Xq'].numpy()), axis=0) 
				mapdata['yq'] = np.concatenate((mapdata.get('yq'), mapdata_['yq'].numpy()), axis=0)

	map_image = mapdata

	# override with a simple test
	# map_array = np.array([[0.9,0.9,0.9,0.9,0.9],
	# 				 [0.9,0,0,0,0.9],
	# 				 [0.9,0,0.5,0,0.9],
	# 				 [0.9,0,0,0,0.9],
	# 				 [0.9,0.9,0.9,0.9,0.9]]
	# 				 )

	#map_image = {}
	#map_image['Xq'] = []
	#map_image['yq'] = []
	#for i in range(len(map_array)):
	#	for j in range(len(map_array[0])):
	#		map_image['Xq'].append([i,j])
	#		map_image['yq'].append(map_array[i][j])
	#map_image['Xq'] = np.array(map_image['Xq'])
	#map_image['yq'] = np.array(map_image['yq'])

	G = nx.Graph()

	pos = nx.get_node_attributes(G, 'pos')
	return G


class GNG():
	"""."""

	def __init__(self, data, eps_b=0.05, eps_n=0.0005, max_age=30,
				 lambda_=10, alpha=0.5, d=0.0005, max_nodes=100):
		"""."""
		self.graph = nx.Graph()
		self.data = data
		# toggle the probabilities
		self.data['yq'] = np.ones(len(self.data['yq'])) - self.data['yq']
		self.data['yq'] = np.exp(10*self.data['yq'])
		# normalize the probabilities
		self.data['yq'] /= np.linalg.norm(self.data['yq'], ord=1)
		self.eps_b = eps_b
		self.eps_n = eps_n
		self.max_age = max_age
		self.lambda_ = lambda_
		self.alpha = alpha
		self.d = d
		self.max_nodes = max_nodes
		self.num_of_input_signals = 0

		self.pos = None
		print(np.random.choice(len(data['Xq']), p=data['yq']))
		node1 = self.data['Xq'][np.random.choice(len(self.data['Xq']), p=self.data['yq'])]
		node2 = self.data['Xq'][np.random.choice(len(self.data['Xq']), p=self.data['yq'])]
		# make sure you dont select same positions
		if node1[0] == node2[0] and node1[1] == node2[1]:
			print("Rerun ---------------> similar nodes selected")
			return None

		# initialize here
		self.count = 0
		self.graph.add_node(self.count, pos=(node1[0], node1[1]), error=0)
		self.count += 1
		self.graph.add_node(self.count, pos=(node2[0], node2[1]), error=0)
		self.graph.add_edge(self.count - 1, self.count, age=0)

	def distance(self, a, b):
		"""Calculate distance between two points."""
		return ((a[0] - b[0])**2 + (a[1] - b[1])**2)

	def determine_2closest_vertices(self, curnode):
		"""Where this curnode is actually the x,y index of the data we want to analyze."""
		self.pos = nx.get_node_attributes(self.graph, 'pos')
		templist = []
		for node, position in iteritems(self.pos):
			dist = self.distance(curnode, position)
			templist.append([node, dist])
		distlist = np.array(templist)

		ind = np.lexsort((distlist[:, 0], distlist[:, 1]))
		distlist = distlist[ind]

		return distlist[0], distlist[1]

	def get_new_position(self, winnerpos, nodepos):
		"""."""
		move_delta = [self.eps_b * (nodepos[0] - winnerpos[0]), self.eps_b * (nodepos[1] - winnerpos[1])]
		newpos = [winnerpos[0] + move_delta[0], winnerpos[1] + move_delta[1]]

		return newpos

	def get_new_position_neighbors(self, neighborpos, nodepos):
		"""."""
		movement = [self.eps_n * (nodepos[0] - neighborpos[0]), self.eps_n * (nodepos[1] - neighborpos[1])]
		newpos = [neighborpos[0] + movement[0], neighborpos[1] + movement[1]]

		return newpos

	def is_outlier_edge(self, test_edge):
		self.pos = nx.get_node_attributes(self.graph, 'pos')
		test_length = self.distance(self.pos[test_edge[0]], self.pos[test_edge[1]])
		all_edge_lengths = []
		for edge in self.graph.edges():
			all_edge_lengths.append(self.distance(self.pos[edge[0]], self.pos[edge[1]]))
		edge_mean, edge_std = np.mean(all_edge_lengths), np.std(all_edge_lengths),
		cut_off = edge_std * 2
		lower, upper = edge_mean - cut_off, edge_mean + cut_off
		#pdb.set_trace()
		bool_ = test_length < lower or test_length > upper
		return bool_


	def update_winner(self, curnode):
		"""."""
		# find nearest unit and second nearest unit
		winner1, winner2 = self.determine_2closest_vertices(curnode)
		winnernode = winner1[0]
		winnernode2 = winner2[0]
		win_dist_from_node = winner1[1]

		errorvectors = nx.get_node_attributes(self.graph, 'error')

		error1 = errorvectors[winner1[0]]
		# update the new error
		newerror = error1 + win_dist_from_node**2
		self.graph.add_node(winnernode, error=newerror)

		# move the winner node towards current node
		self.pos = nx.get_node_attributes(self.graph, 'pos')
		newposition = self.get_new_position(self.pos[winnernode], curnode)
		self.graph.add_node(winnernode, pos=newposition)

		# now update all the neighbors distances and their ages
		neighbors = nx.all_neighbors(self.graph, winnernode)
		age_of_edges = nx.get_edge_attributes(self.graph, 'age')
		for n in neighbors:
			newposition = self.get_new_position_neighbors(self.pos[n], curnode)
			self.graph.add_node(n, pos=newposition)
			key = (int(winnernode), n)
			if key in age_of_edges:
				newage = age_of_edges[(int(winnernode), n)] + 1
			else:
				newage = age_of_edges[(n, int(winnernode))] + 1
			self.graph.add_edge(winnernode, n, age=newage)

		# no sense in what I am writing here, but with algorithm it goes perfect
		# if winnner and 2nd winner are connected, update their age to zero
		if (self.graph.get_edge_data(winnernode, winnernode2) is not None):
			self.graph.add_edge(winnernode, winnernode2, age=0)
		else:
			# else create an edge between them
			self.graph.add_edge(winnernode, winnernode2, age=0)

		# if there are ages more than maximum allowed age, remove them
		age_of_edges = nx.get_edge_attributes(self.graph, 'age')
		for edge, age in iteritems(age_of_edges):

			if age > self.max_age:  # self.is_outlier_edge(edge)
				#pdb.set_trace()
				self.graph.remove_edge(edge[0], edge[1])
				# if it causes isolated vertix, remove that vertex as well

				for node in self.graph.nodes():
					if not self.graph.neighbors(node):
						self.graph.remove_node(node)

	def get_average_dist(self, a, b):
		"""."""
		av_dist = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]

		return av_dist

	def save_img(self, fignum, output_images_dir='images'):
		"""."""
		fig = pl.figure(fignum, figsize=(10,10))
		ax = fig.add_subplot(111)
		pl.scatter(map_image['Xq'][:, 0], map_image['Xq'][:, 1], c=map_image['yq'], cmap='jet', s=100, vmin=0, vmax=1, edgecolors='')
		#pl.scatter(self.data['Xq'][:, 0], self.data['Xq'][:, 1], c=self.data['yq'], cmap='jet', s=2, vmin=0, vmax=(1/41269.54), edgecolors='')
		pl.colorbar()
		#pl.xlim([-40,40]); pl.ylim([-40,40])
		#pl.imshow(map_image, pl.cm.gray)
		# Dont draw G 
		#nx.draw(G, pos, with_labels=False, node_size=40, alpha=.3, width=1.5)
		#nx.draw(G, pos, node_color='#ffffff', with_labels=False, node_size=100, alpha=1.0, width=1.5)
		position = nx.get_node_attributes(self.graph, 'pos')
		nx.draw(self.graph, position, node_color='r', node_size=25, with_labels=False, edge_color='g', width=2.0)
		ages_ = nx.get_edge_attributes(self.graph,'age')
		#pdb.set_trace()
		nx.draw_networkx_edge_labels(self.graph,position,edge_labels=ages_, font_size=7)
		pl.title('Growing Neural Gas')
		pl.savefig("{0}/{1}.png".format(output_images_dir, str(fignum)))

		pl.clf()
		pl.close(fignum)
		self.graph_dump(output_images_dir)


	def samples_plot(self):
		fig = pl.figure("samples_out", figsize=(8,8))
		ax = fig.add_subplot(111)

		pl.scatter(np.array(self.samples)[:,0],np.array(self.samples)[:,1], facecolors='r')
		pl.savefig("samples_intel.png")
		pl.clf()
		pl.close("samples_out")

	def graph_dump(self, output_images_dir):
		with open('./'+ output_images_dir+ '/graph_.pickle', 'wb') as handle:
			pickle.dump(self.graph, handle)

	def train(self, max_iterations=10000, output_images_dir='images_intelmap'):
		"""."""

		if not os.path.isdir(output_images_dir):
			os.makedirs(output_images_dir)

		print("Ouput images will be saved in: {0}".format(output_images_dir))
		fignum = 0
		self.save_img(fignum, output_images_dir)
		self.samples = []
		for i in xrange(1, max_iterations):
			print("Iterating..{0:d}/{1}".format(i, max_iterations))
			iter_list = self.data['Xq'][np.random.choice(len(self.data['Xq']), size=600, p=self.data['yq'])]
			self.samples.extend(iter_list)
			for x in iter_list:
				self.update_winner(x)

				# step 8: if number of input signals generated so far
				if i % self.lambda_ == 0 and len(self.graph.nodes()) <= self.max_nodes:
					# find a node with the largest error
					errorvectors = nx.get_node_attributes(self.graph, 'error')
					import operator
					node_largest_error = max(iteritems(errorvectors), key=operator.itemgetter(1))[0]

					# find a node from neighbor of the node just found,
					# with largest error
					neighbors = self.graph.neighbors(node_largest_error)
					max_error_neighbor = None
					max_error = -1
					errorvectors = nx.get_node_attributes(self.graph, 'error')
					for n in neighbors:
						if errorvectors[n] > max_error:
							max_error = errorvectors[n]
							max_error_neighbor = n

					# insert a new unit half way between these two
					self.pos = nx.get_node_attributes(self.graph, 'pos')

					newnodepos = self.get_average_dist(self.pos[node_largest_error], self.pos[max_error_neighbor])
					self.count = self.count + 1
					newnode = self.count
					self.graph.add_node(newnode, pos=newnodepos)

					# insert edges between new node and other two nodes
					self.graph.add_edge(newnode, max_error_neighbor, age=0)
					self.graph.add_edge(newnode, node_largest_error, age=0)

					# remove edge between the other two nodes

					self.graph.remove_edge(max_error_neighbor, node_largest_error)

					# decrease error variable of other two nodes by multiplying with alpha
					errorvectors = nx.get_node_attributes(self.graph, 'error')
					error_max_node = self.alpha * errorvectors[node_largest_error]
					error_max_second = self.alpha * max_error
					self.graph.add_node(max_error_neighbor, error=error_max_second)
					self.graph.add_node(node_largest_error, error=error_max_node)

					# initialize the error variable of newnode with max_node
					self.graph.add_node(newnode, error=error_max_node)

					fignum += 1
					if i % 100==0:
						self.save_img(fignum, output_images_dir)

				# step 9: Decrease all error variables
				errorvectors = nx.get_node_attributes(self.graph, 'error')
				for i in self.graph.nodes():
					olderror = errorvectors[i]
					newerror = olderror - self.d * olderror
					self.graph.add_node(i, error=newerror)
		self.samples_plot()


def main(map_type):
	"""."""
	global pos, G
	G = read_file_hilbert_maps(map_=map_type)
	inList = []
	for key, value in iteritems(pos):
		inList.append([value[0], value[1]])

	mat = np.array(inList, dtype='float64')
	return copy.deepcopy(map_image)


def sort_nicely(limages):
	"""."""
	def convert(text): return int(text) if text.isdigit() else text

	def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
	limages = sorted(limages, key=alphanum_key)
	return limages


def convert_images_to_gif(output_images_dir, output_gif):
	"""Conveoutput_images_dirrt a list of images to a gif."""

	image_dir = "{0}/*.png".format(output_images_dir)
	list_images = glob.glob(image_dir)
	file_names = sort_nicely(list_images)
	images = [imageio.imread(fn) for fn in file_names]
	imageio.mimsave(output_gif, images)


if __name__ == "__main__":
	#pdb.set_trace()
	map_type = "intel"
	data = main(map_type)
	grng = GNG(data)
	output_images_dir = 'ph_gng_intel_271dense_template'
	output_gif = "gng_intel_271_template.gif"

	if grng is not None:
		grng.train(max_iterations=5000, output_images_dir=output_images_dir)
		convert_images_to_gif(output_images_dir, output_gif)
