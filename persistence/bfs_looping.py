"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import networkx as nx
graph = nx.Graph()
for i in range(8):
	graph.add_node(i)
graph.add_node(0, pos=[-1,2])
graph.add_node(1, pos=[-2,4])
graph.add_node(2, pos=[1,4])
graph.add_node(3, pos=[0,0])
graph.add_node(4, pos=[-1,6])
graph.add_node(5, pos=[3,5])
graph.add_node(6, pos=[2,-1])
graph.add_node(7, pos=[4,0])

graph.add_edge(0,1)
graph.add_edge(0,2)
graph.add_edge(0,3)
graph.add_edge(1,4)
graph.add_edge(2,4)
graph.add_edge(2,5)
graph.add_edge(3,6)
graph.add_edge(5,7)
graph.add_edge(6,7)


class BFSLoopDetection():
	def __init__(self):
		self.parent = {}
		self.loops = {}

	def compute_loop(self, node1, node2, start):
		path1 = [node1]
		while self.parent[node1] != start:
			path1.append(self.parent[node1])
			node1 = self.parent[node1]
		#print("path1", path1, "node1", node1)
		path2 = [node2]
		while self.parent[node2] != start:
			path2.append(self.parent[node2])
			node2 = self.parent[node2]
		#print("path2", path2, "node2", node2)
		path2.reverse()
		return path1 + [start] + path2

	def get_breadth_first_search_loops(self, g, start):
		queue = [start]
		explored = {}
		for node in g.nodes:
			explored[node] = False
		explored[start] = True
		self.parent[start] = None
		pairs_loop_end = []
		loop_count = 0
		while len(queue) != 0:
			node = queue.pop(0)
			for adj in g.neighbors(node):
				if not explored[adj]:
					explored[adj] = True
					queue.append(adj)
					self.parent[adj] = node
				else:
					if self.parent[node] == adj:
						continue
					else:
						# the node is already explored
						if node > adj:
							if (adj, node) not in pairs_loop_end:
								pairs_loop_end.append((adj, node))
								self.loops[loop_count] = self.compute_loop(adj, node, start)
								loop_count += 1
						else:
							if (node, adj) not in pairs_loop_end:
								pairs_loop_end.append((node, adj))
								self.loops[loop_count] = self.compute_loop(node, adj, start)
								loop_count += 1

						#self.loops[loop_count] = self.compute_loop(node, adj, start)
						#loop_count += 1
		return self.loops

import time
t1 = time.time()
loopfinder = BFSLoopDetection()
loops = loopfinder.get_breadth_first_search_loops(graph, 0)
t2 = time.time()
print(loops, t2-t1)

######### write code for inside detection
# store the intersection
feature = [[1, 1]]
segment = [[1, 1], [1, 11]]
import numpy as np
position = nx.get_node_attributes(graph, "pos")
for node1, node2 in graph.edges:
	A = np.array([[segment[1][0] - segment[0][0], -1* (position[node2][0]-position[node1][0])],
				  [segment[1][1] - segment[0][1], -1* (position[node2][1]-position[node1][1])]])
	b = np.array([position[node1][0] - segment[0][0], position[node1][1] - segment[0][1]])
	x = np.linalg.solve(A,b)
	if 0<=x[0]<=1 and 0<=x[1]<=1:
		intersection_edges.append([node1, node2])
		print("true", node1, node2)

for key, cycle in loops.items():
	for i in range(len(cycle)):


