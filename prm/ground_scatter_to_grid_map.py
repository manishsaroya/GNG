"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import pandas as pd
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import pickle

if __name__ == "__main__":
	# load the samples
	g = pd.read_csv("../../Bayesian_Hilbert_Maps/BHM/original/intel.csv", delimiter=',').values
	print('shapes:', g.shape)
	X_train = np.float_(g[:, 0:3])
	Y_train = np.float_(g[:, 3][:, np.newaxis]).ravel()  # * 2 - 1

	# resolution = 3
	df_points = pd.DataFrame({"x": X_train[:, 1], "y": X_train[:, 2], "v": Y_train})
	# x = X_train[:, 1]
	# y = X_train[:, 2]
	# GSIZE = int(resolution*abs(40))
	# GSIZE_y = int(resolution*abs(35))
	# X, Y = np.mgrid[-20:20:GSIZE * 1j, -25:10:GSIZE_y * 1j]
	# grid = np.c_[X.ravel(), Y.ravel()]

	cell_max_min = [-20,20,-25,10]
	q_resolution = 0.3
	xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
						 np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
	X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

	points = np.c_[df_points.x, df_points.y]

	tree = KDTree(X_query)
	dist, indices = tree.query(points) #, distance_upper_bound=q_resolution)
	grid_count = df_points.groupby(indices).v.count()
	grid_values = df_points.groupby(indices).v.sum()

	df_grid = pd.DataFrame(X_query, columns=["x", "y"])
	df_grid["v"] = grid_values / (grid_count - grid_values)

	fig, ax = plt.subplots(figsize=(20, 20))
	#ax.plot(df_points.x, df_points.y, "kx", alpha=0.2)
	for idx, value in enumerate(df_grid.v):
		if not np.isnan(value):
			if value > 0.5:
				df_grid.v[idx] = 1 #int(value>0.5)
			elif (grid_count[idx] - grid_values[idx])>3:
				df_grid.v[idx] = 0
			else:
				df_grid.v[idx] = np.NaN
	df_grid = df_grid.dropna()
	mapper = ax.scatter(df_grid.x, df_grid.y, c=df_grid.v,
						cmap="viridis",
						linewidths=0,
						s=200, marker="o")
	plt.colorbar(mapper, ax=ax)
	plt.show()
	map_data = {"Xq": np.array(list(zip(list(df_grid.x), list(df_grid.y)))), "yq": np.array(list(df_grid.v))}
	with open('ground_map_q_resolution.pickle', 'wb') as handle:
		pickle.dump(map_data, handle)

