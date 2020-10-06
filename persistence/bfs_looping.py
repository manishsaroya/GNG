"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
import numpy as np
import networkx as nx


class BFSLoopDetection:
	def __init__(self):
		self.parent = {}
		self.loops = {}

	def compute_loop(self, node1, node2, start):
		path1 = [node1]
		while self.parent[node1] != start:
			path1.append(self.parent[node1])
			node1 = self.parent[node1]
		# print("path1", path1, "node1", node1)
		path2 = [node2]
		while self.parent[node2] != start:
			path2.append(self.parent[node2])
			node2 = self.parent[node2]
		# print("path2", path2, "node2", node2)
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

						# self.loops[loop_count] = self.compute_loop(node, adj, start)
						# loop_count += 1
		return self.loops


def get_directed_intersection_attribute(g, f):
	segment = [f, [f[0], f[1]+100]]
	position = nx.get_node_attributes(g, "pos")
	H = g.to_directed()
	intersect = {}
	for node1, node2 in H.edges:
		A = np.array([[segment[1][0] - segment[0][0], -1* (position[node2][0]-position[node1][0])],
					  [segment[1][1] - segment[0][1], -1* (position[node2][1]-position[node1][1])]])
		b = np.array([position[node1][0] - segment[0][0], position[node1][1] - segment[0][1]])
		x = np.linalg.solve(A,b)
		if 0 <= x[0] <= 1 and 0 <= x[1] <= 1:
			intersect[(node1, node2)] = True
		else:
			intersect[(node1, node2)] = False
	nx.set_edge_attributes(H, intersect, "intersect")
	return nx.get_edge_attributes(H, "intersect")


def get_polygon_loop(loops_dict, intersect_attr):
	for key, cycle in loops_dict.items():
		intersect_count = 0
		for i in range(len(cycle)):
			if intersect_attr[(cycle[i], cycle[i-1])]:
				intersect_count += 1
		if intersect_count % 2 == 1:
			#print("inside polygon", cycle)
			return True, cycle
	return False, None


if __name__ == "__main__":
	graph = nx.Graph()

	graph.add_node(0, pos=[-1, 2])
	graph.add_node(1, pos=[-2, 4])
	graph.add_node(2, pos=[1, 4])
	graph.add_node(3, pos=[0, 0])
	graph.add_node(4, pos=[-1, 6])
	graph.add_node(5, pos=[3, 5])
	graph.add_node(6, pos=[2, -1])
	graph.add_node(7, pos=[4, 0])

	graph.add_edge(0, 1)
	graph.add_edge(0, 2)
	graph.add_edge(0, 3)
	graph.add_edge(1, 4)
	graph.add_edge(2, 4)
	graph.add_edge(2, 5)
	graph.add_edge(3, 6)
	graph.add_edge(5, 7)
	graph.add_edge(6, 7)

	loopfinder = BFSLoopDetection()
	loops = loopfinder.get_breadth_first_search_loops(graph, 0)
	print(loops)

	######### write code for inside detection
	# store the intersection
	feature = [[-1.1, 3]]
	for f in feature:
		collisions = get_directed_intersection_attribute(graph, f)
		is_inside, polygon = get_polygon_loop(loops, collisions)
		print(is_inside, polygon)




