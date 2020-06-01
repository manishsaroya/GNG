"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import gudhi
import networkx as nx
from utils import load_graph, create_intensity_graph, create_simplex_from_graph, print_complex_attributes
import gudhi
import matplotlib.pyplot as plt


if __name__ == "__main__":
    G = load_graph('./../ph_gng_intel_271dense_template/graph_.pickle')
    G = create_intensity_graph(G, "intel")
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



#count = 0
# G = nx.Graph()
# G.add_node(0, val=1-0.7)
# G.add_node(1, val=1-0.8)
# G.add_node(2, val=1-0.6)
# G.add_node(3, val=1-0.9)
# G.add_node(4, val=1-0.5)
#
# G.add_edge(0,1, val=0.8)
# G.add_edge(1,2, val=0.8)
# G.add_edge(2,3, val=0.9)
# G.add_edge(3,4, val=0.9)
# G.add_edge(3,0, val=0.9)
# G.add_edge(0,4, val=0.7)

# print("Cliques recursive")
# #print(list(nx.enumerate_all_cliques(G)))
# print("######")
