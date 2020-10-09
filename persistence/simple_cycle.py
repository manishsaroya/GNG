"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
N = 100000 #len(local_graph.nodes)
graph_list = [[] for i in range(N)]
cycles = [[] for i in range(N)]

# Function to mark the vertex with
# different colors for different cycles
def dfs_cycle(u, p, color: list,
			mark: list, par: list):
	global cyclenumber

	# already (completely) visited vertex.
	if color[u] == 2:
		return

	# seen vertex, but was not
	# completely visited -> cycle detected.
	# backtrack based on parents to
	# find the complete cycle.
	if color[u] == 1:
		cyclenumber += 1
		cur = p
		mark[cur] = cyclenumber

		# backtrack the vertex which are
		# in the current cycle thats found
		while cur != u:
			cur = par[cur]
			mark[cur] = cyclenumber

		return

	par[u] = p

	# partially visited.
	color[u] = 1

	# simple dfs on graph
	for v in graph_list[u]:

		# if it has not been visited previously
		if v == par[u]:
			continue
		dfs_cycle(v, u, color, mark, par)

	# completely visited.
	color[u] = 2

# add the edges to the graph
def addEdge(u, v):
	graph_list[u].append(v)
	graph_list[v].append(u)

# Function to print the cycles
def printCycles(edges, mark: list):

	# push the edges that into the
	# cycle adjacency list
	for i in range(1, edges + 1):
		if mark[i] != 0:
			cycles[mark[i]].append(i)
	#return cycles

	# print all the vertex with same cycle
	for i in range(1, cyclenumber + 1):

		# Print the i-th cycle
		print("Cycle Number %d:" % i, end = " ")
		for x in cycles[i]:
			print(x, end = " ")
		print()
	return cycles

# Driver Code
for node1, node2 in local_graph.edges:
	addEdge(local_nodes.index(node1), local_nodes.index(node2))

	# add edges
	# addEdge(1, 2)
	# addEdge(2, 3)
	# addEdge(3, 4)
	# addEdge(4, 6)
	# addEdge(4, 7)
	# addEdge(5, 6)
	# addEdge(3, 5)
	# addEdge(7, 8)
	# addEdge(6, 10)
	# addEdge(5, 9)
	# addEdge(10, 11)
	# addEdge(11, 12)
	# addEdge(11, 13)
	# addEdge(12, 13)

# arrays required to color the
# graph, store the parent of node
color = [0] * N
par = [0] * N

# mark with unique numbers
mark = [0] * N

# store the numbers of cycle
cyclenumber = 0
edges = len(local_graph.edges) -1

# call DFS to mark the cycles
dfs_cycle(local_nodes.index(closest_node), local_nodes.index(closest_node), color, mark, par)

# function to print the cycles
cycles = printCycles(edges, mark)
total_path = []
for cycle in cycles:
	total_path.extend([local_nodes[l] for l in cycle])
total_path = [local_nodes[l] for l in cycles[89]]