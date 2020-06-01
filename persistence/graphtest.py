import gudhi
import networkx as nx
import matplotlib.pyplot as plt

count = 0
G = nx.Graph()
G.add_node(0, val=1-0.7)
G.add_node(1, val=1-0.8)
G.add_node(2, val=1-0.6)
G.add_node(3, val=1-0.9)
G.add_node(4, val=1-0.5)

# G.add_edge(0,1, val=0.8)
# G.add_edge(1,2, val=0.8)
# G.add_edge(2,3, val=0.9)
# G.add_edge(3,4, val=0.9)
# G.add_edge(3,0, val=0.9)
# G.add_edge(0,4, val=0.7)

G.add_edge(0,1)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(3,4)
G.add_edge(3,0)
G.add_edge(0,4)

print("Cliques recursive")
#print(list(nx.enumerate_all_cliques(G)))
print("######")


st = gudhi.SimplexTree()
node_values = nx.get_node_attributes(G,"val")
print("node intensitie", node_values)
for clique in nx.enumerate_all_cliques(G):
	clique_value = node_values[clique[0]]
	for n in clique:
		# take max values
		if clique_value < node_values[n]:
			clique_value = node_values[n]
	st.insert(clique,clique_value)


result_str = 'num_vertices=' + repr(st.num_vertices())
print(result_str)
result_str = 'num_simplices=' + repr(st.num_simplices())
print(result_str)
print("skeleton(2) =")
for sk_value in st.get_skeleton(2):
	print(sk_value)

if st.make_filtration_non_decreasing():
	print("modified filtration value")
st.initialize_filtration()
for i in st.persistence():
	print(i)
graph_persistence = st.persistence()
print(graph_persistence)
gudhi.plot_persistence_diagram(graph_persistence, alpha=1, legend=True )
plt.show()
