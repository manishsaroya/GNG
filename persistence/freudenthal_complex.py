"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu

Freudenthal complex creates simplicial complex out of an intensity image
"""
from gudhi import SimplexTree


class FreudenthalComplex:
    def __init__(self, map_array):
        """
        :param map_array: takes in image in array form
        """
        self.height = len(map_array)
        self.width = len(map_array[0])
        self.map_array = map_array
        self.map_index_dict = self.get_map_index_dict()

    def get_map_index_dict(self):
        map_index_dict = {}
        for i in range(self.height):
            for j in range(self.width):
                map_index_dict[i * self.width + j] = self.map_array[i, j]
        return map_index_dict

    def max_value(self, clique):
        clique_value = self.map_index_dict[clique[0]]
        for n in clique:
            # take max values
            if clique_value < self.map_index_dict[n]:
                clique_value = self.map_index_dict[n]
        return clique_value

    def init_freudenthal_2d(self):
        """
        :return: Freudenthal triangulation of 2d grid in gudhi simplex tree form
        """
        width = self.width
        height = self.height

        st = SimplexTree()
        count = 0
        for i in range(height):
            for j in range(width):
                ind = i * width + j

                st.insert([ind], self.max_value([ind]))
                count += 1
        # 1-cells
        # pdb.set_trace()
        for i in range(height):
            for j in range(width - 1):
                ind = i * width + j
                st.insert([ind, ind + 1], self.max_value([ind, ind + 1]))
                count += 1
        # pdb.set_trace()
        for i in range(height - 1):
            for j in range(width):
                ind = i * width + j
                st.insert([ind, ind + width], self.max_value([ind, ind + width]))
                count += 1
        # pdb.set_trace()
        # 2-cells + diagonal 1-cells
        for i in range(height - 1):
            for j in range(width - 1):
                ind = i * width + j
                # diagonal
                st.insert([ind, ind + width + 1], self.max_value([ind, ind + width + 1]))
                count += 1
                # 2-cells
                st.insert([ind, ind + 1, ind + width + 1], self.max_value([ind, ind + 1, ind + width + 1]))
                count += 1
                st.insert([ind, ind + width, ind + width + 1], self.max_value([ind, ind + width, ind + width + 1]))
                count += 1
            # pdb.set_trace()
        print(count, "counts")
        return st
