"""
Author: Manish Saroya
Contact: saroyam@oregonstate.edu

Description: Contains deprecated functions
"""


def read_pgm(filename, byteorder='>'):
    """
    :param filename:
    :param byteorder:
    :return: Return image data from a raw PGM file as numpy array.
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
                            dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                            count=int(width) * int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))
