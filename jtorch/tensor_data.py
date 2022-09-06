import random
from .operators import prod
from numpy import array, float64, ndarray
import numba

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


def index_to_position(index, strides):
    """
    Converts a multidimensional tensor `index` into a single-dimensional position in
    storage based on strides.

    Args:
       index (array-like): index tuple of ints
       strides (array-like): tensor strides

    Return:
        int : position in storage
    """

    # ASSIGN2.1
    position = 0
    for ind, stride in zip(index, strides):
        position += ind * stride
    return position
    # END ASSIGN2.1


def count(position, shape, out_index):
    """
    Convert a `position` to an index in the `shape`.
    Should ensure that enumerating position 0 ... size of a
    tensor produces every index exactly once. It
    may not be the inverse of `index_to_position`.

    Args:
       position (int): current position
       shape (tuple): tensor shape
       out_index (array): the index corresponding to position

    Returns:
       None : Fills in `out_index`.

    """
    # ASSIGN2.1
    cur_pos = position
    for i in range(len(shape) - 1, -1, -1):
        sh = shape[i]
        out_index[i] = int(cur_pos % sh)
        cur_pos = cur_pos // sh
    # END ASSIGN2.1


def broadcast_index(big_index, big_shape, shape, out_index):
    """
    Convert a `big_index` into `big_shape` to a smaller `out_index`
    into `shape` following broadcasting rules. In this case
    it may be larger or with more dimensions than the `shape`
    given. Additional dimensions may need to be mapped to 0 or
    removed.

    Args:
       big_index (array-like): multidimensional index of bigger tensor
       big_shape (array-like): tensor shape of bigger tensor
       shape (array-like): tensor shape of smaller tensor
       out_index (array-like): multidimensional index of smaller tensor

    Returns:
       None : Fills in `out_index`.
    """
    # ASSIGN2.4
    for i, s in enumerate(shape):
        if s > 1:
            out_index[i] = big_index[i + (len(big_shape) - len(shape))]
        else:
            out_index[i] = 0
    # END ASSIGN2.4


def shape_broadcast(shape1, shape2):
    """
    Broadcast two shapes to create a new union shape.

    Args:
       shape1 (tuple): first shape
       shape2 (tuple): second shape

    Returns:
       tuple: broadcasted shape

    """
    # ASSIGN2.4
    a, b = shape1, shape2
    m = max(len(a), len(b))
    c_rev = [0] * m
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    for i in range(m):
        if i >= len(a):
            c_rev[i] = b_rev[i]
        elif i >= len(b):
            c_rev[i] = a_rev[i]
        else:
            c_rev[i] = max(a_rev[i], b_rev[i])
            if a_rev[i] != c_rev[i] and a_rev[i] != 1:
                raise IndexingError("Broadcast failure {a} {b}")
            if b_rev[i] != c_rev[i] and b_rev[i] != 1:
                raise IndexingError("Broadcast failure {a} {b}")
    return tuple(reversed(c_rev))
    # END ASSIGN2.4


def strides_from_shape(shape):
    """Calculates strdes of a tensor from shape attibute.

    Notes (jovsa): Strides are the offset translations
    required to map index of a tensor to a position on
    a contigious array. These offsets can be calculated
    by finding the product of shapes except self.

    working example:
        let shape  = (3, 5, 2)
        then prod(shape) = 30
        General algo:
            1 - [prod(shape)/i for i in shape]
                ex: [30/3, (30/3)/5, ....]
            2 - last dim always == 1 since you don't need an offset

    """
    layout = [1]
    offset = 1
    for s in reversed(shape):
        layout.append(s * offset)
        offset = s * offset
    return tuple(reversed(layout[:-1]))


class TensorData:
    """Contigious data abstraction for a tensor

    Notes (jovsa):
        The TensorData abstraction is datastore  and a translator.
        The underlying data is stored as a 1-D array (np.array).
        This helps with efficient data storage. Furthermore,
        this abstraction also provides two translations functions:
        [1] - `index_to_position`: converts a num (N-D)
        to a 1-D position.
            - Helpers used: self.strides
                - you could do it without self.strides, but you would have to
                redo this calcuation for each dimention. Since this
                calculation does not change for the lifetime of the tensor,
                better to do it once and store it.

        then,
            position = [stride1 * index1 + stride2 * index2 + ... strideN * indexN]



        [2] - `count`: converts a (1-D) position to a N-D location.
            use mod operator from the -1 dim to the 0th dim to get
            location.


        With this capability you can use this 1-D array to store,
        set and get arbitrary N-D tensors.

        Further commentary:
             - A more sloppy way to write this class without the
             two translation functions would be to store 2 maps
             so that that you can get the 1-D position or the
             N-D location. However, this will grow proposional self.size.

    """

    def __init__(self, storage, shape, strides=None):
        if isinstance(storage, ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")
        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self):
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self):
        "Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions."
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a, shape_b):
        return shape_broadcast(shape_a, shape_b)

    def index(self, index):
        if isinstance(index, int):
            index = array([index])
        if isinstance(index, tuple):
            index = array(index)

        # Check for errors
        if index.shape[0] != len(self.shape):
            raise IndexingError(f"Index {index} must be size of {self.shape}.")
        for i, ind in enumerate(index):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {index} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {index} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self):
        lshape = array(self.shape)
        out_index = array(self.shape)
        for i in range(self.size):
            count(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self):
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key):
        return self._storage[self.index(key)]

    def set(self, key, val):
        self._storage[self.index(key)] = val

    def tuple(self):
        return (self._storage, self._shape, self._strides)

    def permute(self, *order):
        """
        Permute the dimensions of the tensor.

        Args:
           order (list): a permutation of the dimensions

        Returns:
           :class:`TensorData`: a new TensorData with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(
            range(len(self.shape))
        ), f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # ASSIGN2.1
        return TensorData(
            self._storage,
            tuple([self.shape[o] for o in order]),
            tuple([self._strides[o] for o in order]),
        )
        # END ASSIGN2.1

    def to_string(self):
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
