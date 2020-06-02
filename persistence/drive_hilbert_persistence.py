"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import matplotlib.pyplot as plt
import gudhi
from utils import load_hilbert_map, convert_map_dict_to_array, print_complex_attributes
from freudenthal_complex import FreudenthalComplex
import numpy as np
import copy


def process_persistence(persist):
    persistence = []

    for p in persist:
        if p[1][0] < 0.5 and p[1][1]>0.5:
            persistence.append(p)
    return persistence


if __name__ == "__main__":
    map_data, resolution = load_hilbert_map(map_type="intel")
    map_array = convert_map_dict_to_array(map_data, resolution)
    # plt.imshow(map_array)
    # plt.show()
    fc = FreudenthalComplex(map_array)
    st = fc.init_freudenthal_2d()
    print_complex_attributes(st)

    if st.make_filtration_non_decreasing():
        print("modified filtration value")
    st.initialize_filtration()
    if len(st.persistence()) <= 10:
        for i in st.persistence():
            print(i)
    #graph_persistence = st.persistence()
    print(st.persistence_intervals_in_dimension(1))
    persistence = process_persistence(st.persistence())
    #gudhi.plot_persistence_diagram(persistence, alpha=0.8, legend=True)
    # find the most relevant persistent in the dictionary
    # Plot top 10 persistence values
    persistence_weightage = []
    persistence_index = []
    for index, value in enumerate(persistence):
        if value[0]==1:
            persistence


    pose = None
    for indx, intensity in enumerate(map_data['yq']):
        for j in range(5):
            print(j)
            p = persistence[j]
            if p[0] == 1:
                if np.isclose(intensity, p[1][1]):
                    pose = map_data["Xq"][indx]
                    plt.plot(pose[0], pose[1], "r*", markersize=12)
                if np.isclose(intensity, p[1][0]):
                    pose = map_data["Xq"][indx]
                    plt.plot(pose[0], pose[1], "bo", markersize=10)
    plt.scatter(map_data["Xq"][:,0],map_data["Xq"][:,1],c=map_data["yq"])
    plt.colorbar()
    plt.show()
    #map_array[int(point[0] * (1/resolution) + 160)][int(point[1] * (1/resolution) + 160)] = map_dict['yq'][indx]
    #return map_array
    #plt.plot(np.where(map_array == persistence[0][1][1]))
    #plt.show()