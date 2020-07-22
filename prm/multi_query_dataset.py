"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
from prm import hilbert_samples
import pickle

if __name__ == "__main__":
    # load map
    with open("ground_map_q_resolution.pickle", 'rb') as tf:
        map_data = pickle.load(tf)

    goal_list = hilbert_samples(map_data.copy(), 30, num_samples=500)
    start_list = hilbert_samples(map_data.copy(), 30, num_samples=500)

    with open("test_samples/" + 'test_data.pickle', 'wb') as handle:
        pickle.dump([goal_list, start_list], handle)