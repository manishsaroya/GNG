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

cell_max_min = [-30, 20, -10, 10]
q_resolution = 0.2
# g = pd.read_csv("../../Bayesian_Hilbert_Maps/BHM/original/fr079.csv", delimiter=',').values
# print('shapes:', g.shape)
# X_train = np.float_(g[:, 0:3])
# Y_train = np.float_(g[:, 3][:, np.newaxis]).ravel()  # * 2 - 1
#
# df_points = pd.DataFrame({"x": X_train[:, 1], "y": X_train[:, 2], "v": Y_train})
xx, yy = np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
					 np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
X_query = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
#
# points = np.c_[df_points.x, df_points.y]


with open("freiburgh_tree.pickle", 'rb') as tf:
	grid_count, grid_values = pickle.load(tf)

#dist, indices = tree.query(points)  # , distance_upper_bound=q_resolution)
#grid_count = df_points.groupby(indices).v.count()
#grid_values = df_points.groupby(indices).v.sum()

df_grid = pd.DataFrame(X_query, columns=["x", "y"])
df_grid["v"] = grid_values / (grid_count)

fig, ax = plt.subplots(figsize=(20, 20))
for idx, value in enumerate(df_grid.v):
	if not np.isnan(value):
		if value > 0.95:
			df_grid.v[idx] = 1  # int(value>0.5)
		elif (grid_count[idx] - grid_values[idx]) > 3:
			df_grid.v[idx] = 0
		else:
			df_grid.v[idx] = np.NaN
df_grid = df_grid.dropna()
mapper = ax.scatter(df_grid.x, df_grid.y, c=df_grid.v,
					cmap="viridis",
					linewidths=0,
					s=70, marker="o")
plt.colorbar(mapper, ax=ax)
plt.show()
map_data = {"Xq": np.array(list(zip(list(df_grid.x), list(df_grid.y)))), "yq": np.array(list(df_grid.v))}
with open('freiburg_ground_map_q_resolution_final.pickle', 'wb') as handle:
	pickle.dump(map_data, handle)
