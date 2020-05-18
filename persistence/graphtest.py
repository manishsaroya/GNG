import gudhi
import networkx as nx
count = 0
G = nx.Graph()
G.add_node(0, val=1-0.7)
G.add_node(1, val=1-0.8)
G.add_node(2, val=1-0.6)
G.add_node(3, val=1-0.9)
G.add_node(4, val=1-0.5)

G.add_edge(0,1, val=0.8)
G.add_edge(1,2, val=0.8)
G.add_edge(2,3, val=0.9)
G.add_edge(3,4, val=0.9)
G.add_edge(3,0, val=0.9)
G.add_edge(0,4, val=0.7)

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



# st = gudhi.SimplexTree()
# st.insert([0], 0.7)
# st.insert([1],0.8)
# st.insert([2],0.6)
# st.insert([3],0.9)
# st.insert([4],0.5)
# st.insert([0,1],0.8)
# st.insert([1,2],0.8)
# st.insert([2,3],0.9)
# st.insert([3,4],0.9)
# st.insert([3,0],0.9)
# st.insert([0,4],0.7)
# st.insert([0,3,4],0.9)
#st.insert([0,1,2,3,4],0.9)
#st.insert([0, 1], filtration):
#print("[0, 1] inserted")
#if st.insert([0, 1, 2], filtration=4.0):
#    print("[0, 1, 2] inserted")
#if st.find([0, 1]):
#    print("[0, 1] found")
result_str = 'num_vertices=' + repr(st.num_vertices())
print(result_str)
result_str = 'num_simplices=' + repr(st.num_simplices())
print(result_str)
print("skeleton(2) =")
for sk_value in st.get_skeleton(2):
    print(sk_value)

if st.make_filtration_non_decreasing():
	print("modefied filtration value")
st.initialize_filtration()
for i in st.persistence():
	print(i)
print(st.persistence())