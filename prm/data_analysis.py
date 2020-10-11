"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
roadmap_types = ["gng", "gng_top", "gng_top_feedback", "prm", "prm_dense", "prm_dense_hilbert"]

# roadmap_types = ["gng_top", "gng"]
data_save_dic = {"gng": "gng_output/", "gng_top": "gng_top_output/", "gng_top_feedback": "gng_top_feedback_output/", "prm": "prm_output/",
                     "prm_dense": "prm_dense_output/", "prm_dense_hilbert": "prm_dense_hilbert_output/"}
for roadmap in roadmap_types:
	if roadmap == "gng":
		with open(data_save_dic[roadmap] + "freiburg_gng1208_200.pickle", 'rb') as tf:
			gng_data = pickle.load(tf)
	elif roadmap == "gng_top":
		with open(data_save_dic[roadmap] + "freiburg_gngtop_1208_200.pickle", 'rb') as tf:
			gngtop_data = pickle.load(tf)
	elif roadmap == "gng_top_feedback":
		with open(data_save_dic[roadmap] + 'freiburg_gngtop_feedback_1208_200.pickle', 'rb') as tf:
			gngtop_feedback_data = pickle.load(tf)
	elif roadmap == "prm":
		with open(data_save_dic[roadmap] + 'freiburg_prm1208.pickle', 'rb') as tf:
			prm_data = pickle.load(tf)
	elif roadmap == "prm_dense":
		with open(data_save_dic[roadmap] + 'freiburg_prmdense_2500.pickle', 'rb') as tf:
			prmdense_data = pickle.load(tf)
	elif roadmap == "prm_dense_hilbert":
		with open(data_save_dic[roadmap] + 'freiburg_prmdense_hilbert4000.pickle', 'rb') as tf:
			prmdense_hilbert_data = pickle.load(tf)


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
gngtop_feedback_path_cost = []
prm_path_cost = []
prmdense_path_cost =[]
prmdense_hilbert_path_cost =[]

gng_node_explored = []
gngtop_node_explored = []
gngtop_feedback_node_explored = []
prm_node_explored = []
prmdense_node_explored = []
prmdense_hilbert_node_explored = []

gng_mismatch_indices = []
gngtop_mismatch_indices = []
index_list = []
spl_gngtop = 0
spl_gngtop_feedback = 0
spl_gng = 0
spl_prm = 0
spl_prmdense = 0
spl_prmdense_hilbert = 0
spl_count = 0
for i in range(len(gng_data[0])):
	if prmdense_data[0][i]!=gng_data[0][i]:
		gng_mismatch_indices.append(i)
	if prmdense_data[0][i]!=gngtop_data[0][i]:
		gngtop_mismatch_indices.append(i)
	if gng_data[0][i]==True and gngtop_feedback_data[0][i]==False and prmdense_data[0][i]==True:
		index_list.append(i)
	#if prmdense_data[0][i]==False:
	#	index_list.append(i)
	if gng_data[0][i] and gngtop_data[0][i] and gngtop_feedback_data[0][i] and prmdense_data[0][i] and prmdense_hilbert_data[0][i]:# and prm_data[0][i]:
	#if prm_data[0][i]==True:
		gng_path_cost.append(gng_data[2][i])
		gngtop_path_cost.append(gngtop_data[2][i])
		gngtop_feedback_path_cost.append(gngtop_feedback_data[2][i])
		#prm_path_cost.append(prm_data[2][i])
		prmdense_path_cost.append(prmdense_data[2][i])
		prmdense_hilbert_path_cost.append(prmdense_hilbert_data[2][i])

		gng_node_explored.append(gng_data[1][i])
		gngtop_node_explored.append(gngtop_data[1][i])
		gngtop_feedback_node_explored.append(gngtop_feedback_data[1][i])
		#prm_node_explored.append(prm_data[1][i])
		prmdense_node_explored.append(prmdense_data[1][i])
		prmdense_hilbert_node_explored.append(prmdense_hilbert_data[1][i])

	# sil score calculation
	if prmdense_data[0][i]:
		if gngtop_data[0][i]:
			spl_gngtop += (prmdense_data[2][i] / max(prmdense_data[2][i], gngtop_data[2][i]))
		if gngtop_feedback_data[0][i]:
			spl_gngtop_feedback += (prmdense_data[2][i] / max(prmdense_data[2][i], gngtop_feedback_data[2][i]))
		if gng_data[0][i]:
			spl_gng += (prmdense_data[2][i] / max(prmdense_data[2][i], gng_data[2][i]))
		if prm_data[0][i]:
			spl_prm += (prmdense_data[2][i] / max(prmdense_data[2][i], prm_data[2][i]))
		if prmdense_data[0][i]:
			spl_prmdense += (prmdense_data[2][i] / max(prmdense_data[2][i], prmdense_data[2][i]))
		if prmdense_hilbert_data[0][i]:
			spl_prmdense_hilbert += (prmdense_data[2][i] / max(prmdense_data[2][i], prmdense_hilbert_data[2][i]))
		spl_count += 1

print("Path cost")
print("prmdense_cost mean:", np.mean(prmdense_path_cost), np.std(prmdense_path_cost))
print("prmdense_hilbert_cost mean:", np.mean(prmdense_hilbert_path_cost), np.std(prmdense_hilbert_path_cost))
print("gngtop_cost mean:", np.mean(gngtop_path_cost), np.std(gngtop_path_cost))
print("gngtop_feedback_cost mean:", np.mean(gngtop_feedback_path_cost), np.std(gngtop_feedback_path_cost))
print("gng_cost mean:", np.mean(gng_path_cost), np.std(gng_path_cost))
#print("prm_cost mean:", np.mean(prm_path_cost), np.std(prm_path_cost))



print("gng_mismatch_indices",len(gng_mismatch_indices))
print("gngtop_mismatch_indices",len(gngtop_mismatch_indices))

print("node_explored")
print("prmdense_node_explored mean:", np.mean(prmdense_node_explored), np.std(prmdense_node_explored))
print("prmdense_hilbert_node_explored mean:", np.mean(prmdense_hilbert_node_explored), np.std(prmdense_hilbert_node_explored))
print("gngtop_node_explored mean:", np.mean(gngtop_node_explored), np.std(gngtop_node_explored))
print("gngtop_feedback_node_explored mean:", np.mean(gngtop_feedback_node_explored), np.std(gngtop_feedback_node_explored))
print("gng_node_explored mean:", np.mean(gng_node_explored), np.std(gng_node_explored))
#print("prm_node_explored mean:", np.mean(prm_node_explored), np.std(prm_node_explored))


print("spl_prmdense ", spl_prmdense/spl_count)
print("spl_prmdense_hilbert", spl_prmdense_hilbert/spl_count)
print("spl_gngtop ", spl_gngtop/spl_count)
print("spl_gngtop_feedback ", spl_gngtop_feedback/spl_count)
print("spl_gng ", spl_gng/spl_count)
print("spl_prm ", spl_prm/spl_count)