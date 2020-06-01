import numpy as np
import gudhi
import pdb
import matplotlib.pyplot as plt

map_array = np.array([[0.9, 0.9, 0.9, 0.9, 0.9],
                      [0.9, 0, 0, 0, 0.9],
                      [0.9, 0, 0.5, 0, 0.9],
                      [0.9, 0, 0, 0, 0.9],
                      [0.9, 0.9, 0.9, 0.9, 0.9]]
                     )
plt.imshow(map_array)
plt.colorbar()
# plt.show()
# map_array = np.array([[0.9,0.9,0.9],
# 					 [0.9,0,0.9],
# 					 [0.9,0.9,0.9]]
# 					 )

map_dict = {}
h = len(map_array)
w = len(map_array[0])
for i in range(h):
    for j in range(w):
        map_dict[i * w + j] = map_array[i, j]
print(map_dict)


def max_value(clique):
    clique_value = map_dict[clique[0]]
    for n in clique:
        # take max values
        if clique_value < map_dict[n]:
            clique_value = map_dict[n]
    return clique_value




def init_freudenthal_2d(width, height):
    """
	Freudenthal triangulation of 2d grid
	"""
	# row-major format
    # 0-cells
    # global st
    st = gudhi.SimplexTree()
    count = 0
    for i in range(height):
        for j in range(width):
            ind = i * width + j

            st.insert([ind], max_value([ind]))
            count += 1
    # 1-cells
    # pdb.set_trace()
    for i in range(height):
        for j in range(width - 1):
            ind = i * width + j
            st.insert([ind, ind + 1], max_value([ind, ind + 1]))
            count += 1
    # pdb.set_trace()
    for i in range(height - 1):
        for j in range(width):
            ind = i * width + j
            st.insert([ind, ind + width], max_value([ind, ind + width]))
            count += 1
    # pdb.set_trace()
    # 2-cells + diagonal 1-cells
    for i in range(height - 1):
        for j in range(width - 1):
            ind = i * width + j
            # diagonal
            st.insert([ind, ind + width + 1], max_value([ind, ind + width + 1]))
            count += 1
            # 2-cells
            st.insert([ind, ind + 1, ind + width + 1], max_value([ind, ind + 1, ind + width + 1]))
            count += 1
            st.insert([ind, ind + width, ind + width + 1], max_value([ind, ind + width, ind + width + 1]))
            count += 1
        # pdb.set_trace()
    print(count, "counts")
    return st


# return s

st = init_freudenthal_2d(w, h)

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
graph_persistence = st.persistence()
gudhi.plot_persistence_diagram(graph_persistence, legend=True)
plt.show()