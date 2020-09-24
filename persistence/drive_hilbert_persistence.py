"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import matplotlib.pyplot as plt
import gudhi
from persistence.utils import load_hilbert_map, convert_map_dict_to_array, print_complex_attributes
from persistence.freudenthal_complex import FreudenthalComplex
import numpy as np
np.set_printoptions(precision=15)
import copy


def process_persistence(persist):
    persistence = []

    for p in persist:
        if p[1][0] < 0.5 and p[1][1]>0.5:
            persistence.append(p)
    return persistence


def get_top_n_persistence_node_location(n, map_type, location_type="death", feature_type=0):
    """
    :param feature_type: 0 for connected components, 1 for loops
    :param n: top number of persistence
    :param map_type: intel or drive
    :param location_type: string representing birth or death
    :return: returns the birth or death persistence node
    """
    if location_type == "death":
        location_type_index = 1
    elif location_type == "birth":
        location_type_index = 0
    else:
        raise ValueError("Invalid location type")

    map_data, resolution = load_hilbert_map(map_type=map_type)
    map_array = convert_map_dict_to_array(map_data, resolution)

    fc = FreudenthalComplex(map_array)
    st = fc.init_freudenthal_2d()
    print_complex_attributes(st)

    if st.make_filtration_non_decreasing():
        print("modified filtration value")
    st.initialize_filtration()
    if len(st.persistence()) <= 10:
        for i in st.persistence():
            print(i)

    first_persistence = st.persistence_intervals_in_dimension(feature_type)
    if feature_type == 0:
        remove_indices = []
        for i in range(len(first_persistence)):
            if first_persistence[i][1] > 0.4:
                remove_indices.append(i)
        first_persistence = np.delete(first_persistence, remove_indices, 0)
        # remove feature ending after 0.4
    life_span = first_persistence[:,1] - first_persistence[:,0]
    winner_index = life_span.argsort()[-n:][::-1]
    print("len winner index ", len(winner_index))
    #print(life_span)
    winner_persistence = first_persistence[winner_index]
    print(winner_persistence, "winner_persistence")
    top_persistence_node = []
    for indx, intensity in enumerate(map_data['yq']):
        for j in range(n):
            p = winner_persistence[j]
            # if np.isclose(intensity, p[1]):
            #     top_persistence_node.append(map_data["Xq"][indx])
            if np.isclose(intensity, p[location_type_index], rtol=1e-10, atol=1e-13):
                top_persistence_node.append(map_data["Xq"][indx])
                print(j, intensity)
    return top_persistence_node, life_span[winner_index]


if __name__ == "__main__":
    # map_data, resolution = load_hilbert_map(map_type="intel")
    # map_array = convert_map_dict_to_array(map_data, resolution)
    # # plt.imshow(map_array)
    # # plt.show()
    # fc = FreudenthalComplex(map_array)
    # st = fc.init_freudenthal_2d()
    # print_complex_attributes(st)
    #
    # if st.make_filtration_non_decreasing():
    #     print("modified filtration value")
    # st.initialize_filtration()
    # if len(st.persistence()) <= 10:
    #     for i in st.persistence():
    #         print(i)
    # #graph_persistence = st.persistence()
    # #print(st.persistence_intervals_in_dimension(1))
    # #persistence = process_persistence(st.persistence())
    # gudhi.plot_persistence_diagram(st.persistence(), alpha=0.8, legend=True)
    # plt.show()
    # first_persistence = []
    # for p in st.persistence():
    #     if p[0]==1:
    #         first_persistence.append(list(p[1]))
    # first_persistence = np.array(first_persistence)
    #
    # #first_persistence = st.persistence_intervals_in_dimension(1)
    # life_span = first_persistence[:,1] - first_persistence[:,0]
    # winner_index = life_span.argsort()[-20:][::-1]
    # #print(life_span)
    # winner_persistence = first_persistence[winner_index]
    # print(winner_persistence, "winner_persistence")
    #
    # pose = None
    # plt.figure(figsize=(8, 8))
    # for indx, intensity in enumerate(map_data['yq']):
    #     for j in range(3):
    #         p = winner_persistence[j]
    #         if np.isclose(intensity, p[1],rtol=1e-10, atol=1e-15):
    #             pose = map_data["Xq"][indx]
    #             plt.plot(pose[0], pose[1], "wo", markersize=10)
    #         if np.isclose(intensity, p[0],rtol=1e-10, atol=1e-15):
    #             pose = map_data["Xq"][indx]
    #             plt.plot(pose[0], pose[1], "y*", markersize=20)
    # plt.scatter(map_data["Xq"][:, 0], map_data["Xq"][:, 1], c=map_data["yq"],cmap="jet")
    # plt.colorbar()
    # plt.show()
    print("printing top *", get_top_n_persistence(3, "intel"))

