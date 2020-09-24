"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
# get the map
from persistence.biased_sampling import get_samples, samples_plot
from persistence.utils import load_hilbert_map
from persistence.drive_hilbert_persistence import get_top_n_persistence_node_location
import numpy as np
from neupy import algorithms, utils
import matplotlib.pyplot as plt
import os
import argparse
import datetime
import pickle


def create_gng(max_nodes, step=0.2, n_start_nodes=2, max_edge_age=50):
	return algorithms.GrowingNeuralGas(
		n_inputs=2,
		n_start_nodes=n_start_nodes,

		shuffle_data=True,
		verbose=True,

		step=step,
		neighbour_step=0.005,

		max_edge_age=max_edge_age,
		max_nodes=max_nodes,

		n_iter_before_neuron_added=100,
		after_split_error_decay_rate=0.5,
		error_decay_rate=0.995,
		min_distance_for_update=0.01,
	)


def draw_image(data, dir, fignum, graph=None, persistence_birthnode=None, persistence_1homnode=None, samples=None, show=True):
	fig = plt.figure(figsize=(40/4, 35/4))
	plt.axis("equal")
	#plt.style.use('seaborn-dark')
	plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap="jet", s=70, vmin=0, vmax=1, edgecolors='')
	#plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], s=70, vmin=0, vmax=1, edgecolors='')
	plt.colorbar(fraction= 0.047, pad=0.02)
	if graph is not None:
		for node_1, node_2 in graph.edges:
			weights = np.concatenate([node_1.weight, node_2.weight])
			line, = plt.plot(*weights.T, color='lightsteelblue')
			plt.setp(line, linewidth=2, color='lightsteelblue')

	if samples is not None:
		plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1], marker='v', facecolors='cyan')

	# if persistence_birthnode is not None:
	# 	for i in range(len(persistence_birthnode)):
	# 		plt.plot(persistence_birthnode[i][0], persistence_birthnode[i][1], "b*", markersize=25)
	#
	# if persistence_1homnode is not None:
	# 	for i in range(len(persistence_1homnode)):
	# 		plt.plot(persistence_1homnode[i][0], persistence_1homnode[i][1], "r*", markersize=25)

	if show:
		plt.savefig(dir + "graph{}.png".format(fignum))
	# plt.show()


def normalize(data, exp_factor=20):
	# toggle the probabilities
	data['yq'] = np.ones(len(data['yq'])) - data['yq']
	data['yq'] = np.exp(exp_factor * data['yq'])
	# normalize the probabilities
	data['yq'] /= np.linalg.norm(data['yq'], ord=1)
	return data


def plot_loss(train_error_mean, train_error_std, dir):
	plt.figure(figsize=(5, 5))
	plt.plot(train_error_mean, label="train error", color="g")
	plt.fill_between(range(len(train_error_mean)), np.array(train_error_mean) - np.array(train_error_std),
					 np.array(train_error_mean) + np.array(train_error_std), \
					 color="g", label="std", alpha=0.1)
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Error")
	# plt.xlim(0, 250)
	# plt.yticks(np.arange(60, 110, step=10))
	plt.title("Train Error")
	plt.savefig(dir + "training_error.png")

def get_multi_gauss_samples(local_map_data, persistence_loc):
	sample_list = []
	for i in range(2):
		small_list = get_samples(local_map_data.copy(),
								 persistence_loc[np.random.randint(0, len(persistence_loc))],
								 scale=1.5, num_samples=int(600 / 2))
		sample_list.extend(small_list)
	return sample_list


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--runs', type=int, default=1)
	parser.add_argument('--exp_factor', type=int, default=30)
	parser.add_argument('--max_edge_age', type=int, default=20)
	parser.add_argument('--max_epoch', type=int, default=300)
	parser.add_argument('--max_nodes', type=int, default=2000)
	parser.add_argument('--log_dir', type=str, default='./output')
	parser.add_argument('--top_n_persistence', type=int, default=25)
	parser.add_argument('--is_bias_sampling', type=bool, default=False)
	parser.add_argument('--bias_ratio', type=float, default=0.75)
	parser.add_argument('--map_type', type=str, default="freiburg")
	args = parser.parse_args()

	args.log_dir = './output/exp_factor-' + args.map_type + str(args.exp_factor) + "-is-bias-sampling-" + \
				   str(args.is_bias_sampling) + "-bias_ratio-" + str(args.bias_ratio) +"-max_epoch-" + \
				   str(args.max_epoch) + "-max_edge_age-" + str(args.max_edge_age) + "-date-" + \
				   datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '/'

	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	data, resolution = load_hilbert_map(map_type=args.map_type)
	persistence_birth_nodes, persistence_weights = get_top_n_persistence_node_location(args.top_n_persistence, args.map_type,
																  location_type="death", feature_type=0)
	persistence_1hom_nodes, persistence_1hom_weights = get_top_n_persistence_node_location(args.top_n_persistence, args.map_type,
																  location_type="death", feature_type=1)
	#persistence_weights /= np.linalg.norm(persistence_weights, ord=1)
	# iter_list = get_samples(data.copy(), persistence[2], scale=2, num_samples=600)
	# samples_plot(samples)
	original_data = data.copy()
	data = normalize(data, args.exp_factor)

	# hello world
	draw_image(original_data, args.log_dir, 2, persistence_birthnode=persistence_birth_nodes, \
			   persistence_1homnode=persistence_1hom_nodes, show=True)
	#exit()
	# GNG learning code
	utils.reproducible()
	gng = create_gng(args.max_nodes, max_edge_age=args.max_edge_age)
	all_samples = []
	train_error_mean = []
	train_error_std = []
	for epoch in range(args.max_epoch + 1):
		# if epoch / args.max_epoch >= 0.9:
		# 	sample_list = get_samples(original_data.copy(), persistence_birth_nodes[epoch%10], scale=1.5, num_samples=600)
		# else:
		if args.is_bias_sampling and epoch > 70:
			if np.random.uniform(0, 1) < args.bias_ratio:
				sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]
			else:
				print("biasing epoch", epoch)
				if np.random.uniform(0,1) > 0.5:
					sample_list = get_multi_gauss_samples(original_data.copy(), persistence_birth_nodes)
				else:
					sample_list = get_multi_gauss_samples(original_data.copy(), persistence_1hom_nodes)
		else:
			sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]

		#all_samples.extend(sample_list)
		gng.train(sample_list, epochs=1)
		train_error_mean.append(np.mean(gng.errors.train))
		train_error_std.append(np.std(gng.errors.train))
		if epoch % 100 == 0:
			draw_image(original_data, args.log_dir, epoch, graph=gng.graph, persistence_birthnode=persistence_birth_nodes, \
					   persistence_1homnode=persistence_1hom_nodes, show=True)
			with open(args.log_dir + 'gng{:d}.pickle'.format(epoch), 'wb') as handle:
				pickle.dump(gng, handle)

	plot_loss(train_error_mean, train_error_std, args.log_dir)


# Garbage

def draw_persistence(persistence_birthnode, show=False):
	"""
	:param persistence_birthnode:
	:param show:
	:return:
	"""
	# hard coded
	for i in range(len(persistence_birthnode)):
		plt.plot(persistence_birthnode[i][0], persistence_birthnode[i][1], "y*", markersize=20)
	if show:
		plt.show()
	# all_samples = []
	# # correction epochs
	# for epoch in range(len(persistence_birth_nodes)):
	# 	sample_list = get_samples(original_data.copy(), persistence_birth_nodes[epoch], scale=1.5, num_samples=600)
	# 	all_samples.extend(sample_list)
	# 	gng.train(sample_list, epochs=1)
	# 	train_error_mean.append(np.mean(gng.errors.train))
	# 	train_error_std.append(np.std(gng.errors.train))
	#
	# 	draw_image(original_data, gng.graph, args.log_dir, epoch, persistence_birthnode=persistence_birth_nodes,\
	# 			   samples=sample_list, show=True)
	# 	with open(args.log_dir + 'gng{:d}.pickle'.format(epoch), 'wb') as handle:
	# 		pickle.dump(gng, handle)
