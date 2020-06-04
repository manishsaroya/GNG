"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import gudhi
import networkx as nx
from utils import load_graph, create_intensity_graph, create_simplex_from_graph, print_complex_attributes
import gudhi
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    G = load_graph('./../ph_gng_intel_908dense_template/graph_7.pickle')
    G, map_data = create_intensity_graph(G, "intel")
    intensities = nx.get_node_attributes(G, "intensity")
    st = create_simplex_from_graph(G)
    print("position",nx.get_node_attributes(G,"pos"))
    print_complex_attributes(st)

    if st.make_filtration_non_decreasing():
        print("modified filtration value")
    st.initialize_filtration()
    if len(st.persistence()) <= 20:
        for i in st.persistence():
            print(i)
    graph_persistence = st.persistence()
    gudhi.plot_persistence_diagram(graph_persistence, alpha=0.8, legend=True)
    plt.show()

    first_persistence = st.persistence_intervals_in_dimension(1)
    life_birth = first_persistence[:, 0]
    winner_index = life_birth.argsort()[:20]
    print(life_birth)
    winner_persistence = first_persistence[winner_index]
    # mapdata will not be required.
    pose = None
    position = nx.get_node_attributes(G, "pos")
    plt.figure(figsize=(10, 10))
    for indx, intensity in intensities.items():
        for j in range(20):
            #print(j)
            p = winner_persistence[j]
            if np.isclose(intensity, p[1]):
                pose = position[indx]
                plt.plot(pose[0], pose[1], "wo", markersize=10)
            if np.isclose(intensity, p[0]):
                pose = position[indx]
                plt.plot(pose[0], pose[1], "y*", markersize=20)
    plt.scatter(map_data["Xq"][:,0],map_data["Xq"][:,1],c=map_data["yq"], cmap="jet")
    plt.colorbar()
    #plt.show()

    # plotting the graph
    position = nx.get_node_attributes(G, 'pos')
    nx.draw(G, position, node_color='r', node_size=25, with_labels=False, edge_color='g', width=2.0)
    #ages_ = nx.get_edge_attributes(self.graph, 'age')
    # pdb.set_trace()
    #nx.draw_networkx_edge_labels(self.graph, position)
    plt.title('Growing Neural Gas')
    #pl.savefig("{0}/{1}.png".format(output_images_dir, str(fignum)))
    plt.show()