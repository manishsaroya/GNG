"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import matplotlib.pyplot as plt
import gudhi
from utils import load_hilbert_map, convert_map_dict_to_array, print_complex_attributes
from freudenthal_complex import FreudenthalComplex


if __name__ == "__main__":
    map_data = load_hilbert_map(map_type="intel")
    map_array = convert_map_dict_to_array(map_data)
    plt.imshow(map_array)
    plt.show()
    fc = FreudenthalComplex(map_array)
    st = fc.init_freudenthal_2d()
    print_complex_attributes(st)

    if st.make_filtration_non_decreasing():
        print("modified filtration value")
    st.initialize_filtration()
    if len(st.persistence()) <= 10:
        for i in st.persistence():
            print(i)
    graph_persistence = st.persistence()
    gudhi.plot_persistence_diagram(graph_persistence, alpha=0.8, legend=True)
    plt.show()