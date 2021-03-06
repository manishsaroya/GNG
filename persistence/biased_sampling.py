"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""

import scipy.stats as st
import numpy as np
from scipy.stats import multivariate_normal
from persistence.utils import load_hilbert_map, convert_map_dict_to_array
import matplotlib.pyplot as plt

# from scipy.stats import multivariate_normal
# x = np.linspace(0, 5, 10, endpoint=False)
# # y = multivariate_normal.pdf(x, mean=2.5, cov=0.5); y
#
#
# plt.plot(x, y)
# plt.show()
# x, y = np.mgrid[-1:1:.01, -1:1:.01]
# pos = np.empty(x.shape + (2,))
# pos[:, :, 0] = x; pos[:, :, 1] = y
# rv = multivariate_normal([.0, .0], [[1.0, 0.], [0., 1.0]])
# plt.contourf(x, y, rv.pdf(pos))
# plt.show()

#def g(x):

def samples_plot(data, samples, fignum, dir, show=False):
	plt.clf()
	fig = plt.figure("samples_out", figsize=(40/2, 35/2))
	plt.axis("equal")
	plt.style.use('seaborn-dark')
	plt.scatter(data['Xq'][:, 0], data['Xq'][:, 1], c=data['yq'], s=10, vmin=0, vmax=1, edgecolors='')
	plt.colorbar(fraction= 0.047, pad=0.02)
	plt.xlim(-30,20)
	plt.ylim(-10,10)
	plt.scatter(np.array(samples)[:, 0], np.array(samples)[:, 1], facecolors='r')
	if show:
		plt.show()
	else:
		plt.savefig(dir + "samples{}.png".format(fignum))
	plt.clf()

def get_samples(map_data, pose, exp_factor, obs_threshold, scale=1, num_samples=600):
	"""
	:param exp_factor: exponential factor
	:param map_data: map info
	:param pose: mean for the multivariate gaussian
	:param scale: std dev
	:param num_samples: number of samples required
	:return: samples
	"""
	rv = multivariate_normal(pose, [[scale**2, 0.], [0., scale**2]])

	map_data['yq'] = np.ones(len(map_data['yq'])) - map_data['yq']
	map_data["yq"][map_data["yq"] < (1-obs_threshold)] = 0
	#map_data['yq'] = np.exp(exp_factor * map_data['yq'])

	for index in range(len(map_data["yq"])):
		map_data["yq"][index] *= rv.pdf(map_data["Xq"][index])

	# normalize the probabilities
	map_data['yq'] /= np.linalg.norm(map_data['yq'], ord=1)

	samples_list = map_data['Xq'][np.random.choice(len(map_data['Xq']), size=num_samples, p=map_data['yq'])]
	return samples_list




if __name__=="__main__":
	map_data_, resolution = load_hilbert_map(map_type="drive")
	samples = get_samples(map_data_, [0, 0], scale=15, num_samples=6000)
	samples_plot(samples)

	# pose = [[0, 40], [0, 0]]
	# rvlist = []
	# for p in pose:
	# 	rvlist.append(multivariate_normal(mean=p, cov=[[225.0, 0.], [0., 225.0]]))
	# #rv = multivariate_normal([.0, 40.0], [[225.0, 0.], [0., 225.0]])
	# map_data['yq'] = np.ones(len(map_data['yq'])) - map_data['yq']
	#
	# map_data['yq'] = np.exp(20 * map_data['yq'])
	# for index in range(len(map_data["yq"])):
	# 	map_data["yq"][index] *= rv.pdf(map_data["Xq"][index])
	# # normalize the probabilities
	# map_data['yq'] /= np.linalg.norm(map_data['yq'], ord=1)
	#
	# ter_list = map_data['Xq'][np.random.choice(len(map_data['Xq']), size=6000, p=map_data['yq'])]
	# samples_plot(ter_list)
	# #map_array = convert_map_dict_to_array(map_data, resolution)
	# #plt.imshow(map_array)
	# plt.show()