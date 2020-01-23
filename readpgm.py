import pdb
# def read_pgm(pgmf):
#     """Return a raster of integers from a PGM as a list of lists."""
#     #print(pgmf.readline())
#     assert pgmf.readline() == b'P5\n'
#     pdb.set_trace()
#     (width, height) = [int(i) for i in pgmf.readline().split()]
#     depth = int(pgmf.readline())
#     assert depth <= 255

#     raster = []
#     for y in range(height):
#         row = []
#         for y in range(width):
#             row.append(ord(pgmf.read(1)))
#         raster.append(row)
#     return raster

# f = open('explore_gmapping.pgm', 'rb')
# read_pgm(f)

import re
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


if __name__ == "__main__":
    from matplotlib import pyplot
    image = read_pgm("explore_gmapping.pgm", byteorder='<')
    pdb.set_trace()
    print(image)
    pyplot.imshow(image, pyplot.cm.gray)
    pyplot.show()