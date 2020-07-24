"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
roadmap_types = ["gng", "gng_top", "prm", "prm_dense"]
# roadmap_types = ["gng_top", "gng"]
data_save_dic = {"gng": "gng_output/", "gng_top": "gng_top_output/", "prm": "prm_output/",
				 "prm_dense": "prm_dense_output/"}

for roadmap in roadmap_types:
	if roadmap == "gng":
		with open(data_save_dic[roadmap] + "gng1208_200.pickle", 'rb') as tf:
			gng_data = pickle.load(tf)
	elif roadmap == "gng_top":
		with open(data_save_dic[roadmap] + "gngtop_1208_200.pickle", 'rb') as tf:
			gngtop_data = pickle.load(tf)
	elif roadmap == "prm":
		with open(data_save_dic[roadmap] + 'prm1208.pickle', 'rb') as tf:
			prm_data = pickle.load(tf)
	elif roadmap == "prm_dense":
		with open(data_save_dic[roadmap] + 'prmdense_2500.pickle', 'rb') as tf:
			prmdense_data = pickle.load(tf)


# with open("test_output/prm1208.pickle", 'rb') as tf:
# 	prm_data = pickle.load(tf)
#
# with open("test_output/prmdense_2500.pickle", 'rb') as tf:
# 	prmdense_data = pickle.load(tf)
	# with open("test_output/" + 'prm1208.pickle', 'wb') as handle:
	# 	pickle.dump([success_list, node_explored_list, distance_to_goal_list], handle)
print("hello")
count = 0
gng_path_cost = []
gngtop_path_cost = []
prm_path_cost = []
prmdense_path_cost =[]

gng_node_explored = []
gngtop_node_explored = []
prm_node_explored = []
prmdense_node_explored = []
gng_mismatch_indices = []
gngtop_mismatch_indices = []
index_list = []
for i in range(len(gng_data[0])):
	if prmdense_data[0][i]!=gng_data[0][i]:
		gng_mismatch_indices.append(i)
	if prmdense_data[0][i]!=gngtop_data[0][i]:
		gngtop_mismatch_indices.append(i)
	#if gng_data[0][i]==True and gngtop_data[0][i]==False and prmdense_data[0][i]==True:
	#	index_list.append(i)
	if prmdense_data[0][i]==False:
		index_list.append(i)
	#if gng_data[0][i] and gngtop_data[0][i] and prmdense_data[0][i] and prm_data[0][i]:
	if prm_data[0][i]==True:
		#gng_path_cost.append(gng_data[2][i])
		#gngtop_path_cost.append(gngtop_data[2][i])
		prm_path_cost.append(prm_data[2][i])
		#prmdense_path_cost.append(prmdense_data[2][i])

		#gng_node_explored.append(gng_data[1][i])
		#gngtop_node_explored.append(gngtop_data[1][i])
		prm_node_explored.append(prm_data[1][i])
		#prmdense_node_explored.append(prmdense_data[1][i])

print("Path cost")
print("gngtop_cost mean:", np.mean(gngtop_path_cost), np.std(gngtop_path_cost))
print("gng_cost mean:", np.mean(gng_path_cost), np.std(gng_path_cost))
print("prm_cost mean:", np.mean(prm_path_cost), np.std(prm_path_cost))
print("prmdense_cost mean:", np.mean(prmdense_path_cost), np.std(prmdense_path_cost))
print("gng_mismatch_indices",len(gng_mismatch_indices))
print("gngtop_mismatch_indices",len(gngtop_mismatch_indices))
print("node_explored")
print("gngtop_node_explored mean:", np.mean(gngtop_node_explored), np.std(gngtop_node_explored))
print("gng_node_explored mean:", np.mean(gng_node_explored), np.std(gng_node_explored))
print("prm_node_explored mean:", np.mean(prm_node_explored), np.std(prm_node_explored))
print("prmdense_node_explored mean:", np.mean(prmdense_node_explored), np.std(prmdense_node_explored))