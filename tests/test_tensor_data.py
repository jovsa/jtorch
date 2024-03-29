import jtorch
from hypothesis import given
from hypothesis.strategies import data
from .strategies import tensor_data, indices
import pytest
from pdb import set_trace


@pytest.mark.task2_1
def test_layout():
    "Test basis properties of layout and strides"
    rows, cols = 3, 5
    data = [0] * rows * cols
    tensor_data = jtorch.TensorData(data, (rows, cols), (5, 1))

    assert tensor_data.is_contiguous()
    assert tensor_data.shape == (rows, cols)

    prev = -1
    seen = set()
    for i in range(rows):
        for j in range(cols):
            curr_idx = tensor_data.index((i, j))
            assert curr_idx > prev and curr_idx not in seen
            prev = curr_idx
            seen.add(curr_idx)

    assert tensor_data.index((1, 0)) == 5
    assert tensor_data.index((1, 2)) == 7

    tensor_data = jtorch.TensorData(data, (5, 3), (1, 5))
    assert tensor_data.shape == (5, 3)
    assert not tensor_data.is_contiguous()

    data = [0] * 4 * 2 * 2
    tensor_data = jtorch.TensorData(data, (4, 2, 2))
    assert tensor_data.strides == (4, 2, 1)


@pytest.mark.task2_1
@given(tensor_data())
def test_enumeration(tensor_data):
    "Test enumeration of tensor_datas."
    indices = list(tensor_data.indices())

    # Check that enough positions are enumerated.
    assert len(indices) == tensor_data.size

    # Check that enough positions are enumerated only once.
    assert len(set(tensor_data.indices())) == len(indices)

    # Check that all indices are within the shape.
    for ind in tensor_data.indices():
        for i, p in enumerate(ind):
            assert p >= 0 and p < tensor_data.shape[i]


@pytest.mark.task2_1
@given(tensor_data())
def test_index(tensor_data):
    "Test enumeration of tensor_data."
    # Check that all indices are within the size.
    for ind in tensor_data.indices():
        pos = tensor_data.index(ind)
        assert pos >= 0 and pos < tensor_data.size

    base = [0] * tensor_data.dims
    with pytest.raises(jtorch.IndexingError):
        base[0] = -1
        tensor_data.index(tuple(base))

    if tensor_data.dims > 1:
        with pytest.raises(jtorch.IndexingError):
            base = [0] * (tensor_data.dims - 1)
            tensor_data.index(tuple(base))


@pytest.mark.task2_1
@given(data())
def test_permute(data):
    td = data.draw(tensor_data())
    ind = data.draw(indices(td))
    td_rev = td.permute(*list(reversed(range(td.dims))))
    assert td.index(ind) == td_rev.index(tuple(reversed(ind)))

    td2 = td_rev.permute(*list(reversed(range(td_rev.dims))))
    assert td.index(ind) == td2.index(ind)


@pytest.mark.task2_4
def test_shape_broadcast():
    c = jtorch.shape_broadcast((1,), (5, 5))
    assert c == (5, 5)

    c = jtorch.shape_broadcast((5, 5), (1,))
    assert c == (5, 5)

    c = jtorch.shape_broadcast((1, 5, 5), (5, 5))
    assert c == (1, 5, 5)

    c = jtorch.shape_broadcast((5, 1, 5, 1), (1, 5, 1, 5))
    assert c == (5, 5, 5, 5)

    with pytest.raises(jtorch.IndexingError):
        c = jtorch.shape_broadcast((5, 7, 5, 1), (1, 5, 1, 5))
        print(c)

    with pytest.raises(jtorch.IndexingError):
        c = jtorch.shape_broadcast((5, 2), (5,))
        print(c)

    c = jtorch.shape_broadcast((2, 5), (5,))
    assert c == (2, 5)


@given(tensor_data())
def test_string(tensor_data):
    tensor_data.to_string()
