import sys
import tables
import numpy as np

class Track(object):
    """This class represents a data track in the GenomeDB database.
    This is an abstraction over a single HDF5 file and allows for easy
    retrieval of a numpy array or DNA sequence string for a set of
    chromosomal coordinates. Normally a Track object is obtained by
    calling the open_track or create_track method of the GenomeDB
    object.

    In theory this class could be exteded to allow for a mixture of
    file types (e.g. bigWig, XB or bam) to be accessible from the
    database, although I'm not certain this would be a good idea.
    """
    def __init__(self, name, path, mode="r"):
        self.name = name
        self.path = path
        self.h5f = tables.open_file(path, mode)

        self._missing_chrom = set([])
        
    def __enter__(self):
        sys.stderr.write("Track %s opened\n" % self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr.write("Cleaning up track %s\n" % self.name)
        self.h5f.close()
        return False
    
    def has_chromosome(self, chrom):
        """Returns True if this track contains a particular chromosome,
        False otherwise"""
        chrom_str = str(chrom)
        return chrom_str in self.h5f.root

    def get_array(self, chrom):
        """returns an PyTables ArrayNode for a particular chromosome"""
        node_name = "/" + str(chrom)

        if node_name in self.h5f:
            array_node = self.h5f.get_node(node_name)
        else:
            if str(chrom) not in self._missing_chrom:
                sys.stderr.write("WARNING: track '%s' is missing chromosome '%s'\n" %
                                 (self.name, str(chrom)))
                self._missing_chrom.add(str(chrom))
            return None

        return array_node
    
    def get_val(self, chrom, pos):
        """returns the value of the track at the specified "
        genomic position"""

        array = self.get_array(chrom)

        if array:
            return array[pos-1]

        return np.nan


    def close(self):
        """Closes this track by closing the underlying HDF5 file"""
        self.h5f.close()
    
    def __get_np_slice(self, array_node, start, end):
        """Helper function, gets a numpy array slice corresponding
        to the provided coordinates from a PyTable ArrayNode"""

        if start > end:
            raise ValueError("start (%d) must be <= end (%d)")

        if (start is None) and (end is None):
            return array_node[:]

        if start < 1:
            raise ValueError("start must be >= 1")

        if start is None:
            start_idx = 0
        else:
            start_idx = start-1

        if end is None:
            end_idx = array_node.shape[0]
        else:
            if end > array_node.shape[0]:
                raise ValueError("end (%d) is greater than chromosome "
                                 "length (%d)" % (end, array_node.shape[0]))
            end_idx = end

        return array_node[start_idx:end_idx]
    
    def get_nparray(self, chrom, start=None, end=None):
        """Returns a numpy array of data for the specified chromosome
        or chromosomal region"""
        array = self.get_array(chrom)

        if array is None:
            # default to array of nan
            if hasattr(chrom, "length"):
                array = np.empty(chrom.length, dtype=np.float32)
            else:
                raise ValueError("cannot create array for missing chromosome "
                                 "of unknown length for track '%s'" % self.name)

            array[:] = np.nan

        return self.__get_np_slice(array, start, end)
    
    def get_seq_str(self, chrom, start=None, end=None):
        """Returns a string of sequence of the specified chromosome
        or chromosomal region. It only makes sense to call this function
        for tracks represent sequence data as 8-bit integers that can be
        converted to printable characters."""
        def from_nparray(vals):
            """converts a numpy array into a sequence string"""
            return "".join(chr(x) for x in vals)
        vals = self.get_nparray(chrom, start, end)
        return from_nparray(vals)
