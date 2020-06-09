"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
# get the map
from persistence.biased_sampling import get_samples
from persistence.utils import load_hilbert_map
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


def draw_image(data, graph, dir, fignum, show=True):
	plt.figure(figsize=(15, 15))
	plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], cmap='jet', s=100, vmin=0, vmax=1, edgecolors='')
	plt.colorbar()
	for node_1, node_2 in graph.edges:
		weights = np.concatenate([node_1.weight, node_2.weight])
		line, = plt.plot(*weights.T, color='lightsteelblue')
		plt.setp(line, linewidth=2, color='lightsteelblue')

	plt.xticks([], [])
	plt.yticks([], [])

	if show:
		plt.savefig(dir + "graph{}.eps".format(fignum))
		#plt.show()

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
	plt.fill_between(range(len(train_error_mean)), np.array(train_error_mean)-np.array(train_error_std), np.array(train_error_mean)+np.array(train_error_std),\
					 color="g", label="std", alpha=0.1)
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Error")
	#plt.xlim(0, 250)
	#plt.yticks(np.arange(60, 110, step=10))
	plt.title("Train Error")
	plt.savefig(dir+"training_error.png")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--runs', type=int, default=1)
	parser.add_argument('--exp_factor',type=int, default=20)
	parser.add_argument('--max_epoch', type=int, default=500)
	parser.add_argument('--log_dir', type=str, default='./output')
	args = parser.parse_args()

	args.log_dir = './output/exp_factor-' + str(args.exp_factor) + "-max_epoch-" +\
				   str(args.max_epoch) + "-date-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '/'

	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	data, resolution = load_hilbert_map(map_type="intel")
	original_data = data.copy()
	data = normalize(data, args.exp_factor)

	# GNG learning code
	utils.reproducible()
	gng = create_gng(2000)

	train_error_mean = []
	train_error_std = []
	for epoch in range(args.max_epoch+1):
		sample_list = data['Xq'][np.random.choice(len(data['Xq']), size=600, p=data['yq'])]
		gng.train(sample_list, epochs=1)
		train_error_mean.append(np.mean(gng.errors.train))
		train_error_std.append(np.std(gng.errors.train))
		if epoch % 100 == 0:
			draw_image(original_data, gng.graph, args.log_dir, epoch, True)
			with open(args.log_dir + 'gng{:d}.pickle'.format(epoch), 'wb') as handle:
				pickle.dump(gng, handle)
	plot_loss(train_error_mean,train_error_std, args.log_dir)

