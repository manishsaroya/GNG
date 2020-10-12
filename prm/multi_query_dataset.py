"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
from prm import hilbert_samples
import pickle
from persistence.utils import load_hilbert_map
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # load map
    #with open("freiburg_ground_map_q_resolution_final.pickle", 'rb') as tf:
    # with open("freiburg_ground_map_q_resolution_final.pickle", 'rb') as tf:
    #     map_data = pickle.load(tf)
    # resolution = 0.3
    obstacle_threshold = 0.25
    map_data, resolution = load_hilbert_map("intel")
    #map_data["yq"] = 1.0 * (map_data["yq"] > 0.45)
    fig = plt.figure(figsize=(40 / 4, 35 / 4))
    plt.axis("equal")
    #plt.style.use('seaborn-dark')
    plt.scatter(map_data['Xq'][:, 0], map_data['Xq'][:, 1], c=map_data['yq'], cmap="jet", s=(70/0.3) * resolution*0.2, vmin=0, vmax=1, edgecolors='')
    #plt.scatter(map_data['Xq'][:, 0], map_data['Xq'][:, 1], c=map_data['yq'], s=10, vmin=0, vmax=1, edgecolors='')
    plt.colorbar(fraction=0.047, pad=0.02)
    plt.show()
    goal_list = hilbert_samples(map_data.copy(), 10, obstacle_threshold, num_samples=500)
    start_list = hilbert_samples(map_data.copy(), 10, obstacle_threshold, num_samples=500)

    with open("test_samples/" + 'intel_crap.pickle', 'wb') as handle:
        pickle.dump([goal_list, start_list], handle)