from jtorch import operators
import numpy as np
from hypothesis import given
from hypothesis.strategies import lists
from .strategies import small_floats, assert_close

## correctness testing


@given(small_floats, small_floats)
def test_add_and_mul(x, y):
    assert_close(operators.mul(x, y), x * y)
    assert_close(operators.add(x, y), x + y)
    assert_close(operators.neg(x), -x)
    assert_close(operators.id(x), x)
    assert_close(operators.add(x, y), x + y)
    assert_close(operators.max(x, y), max(x, y))
    assert_close(operators.sigmoid(x), 1 / (1 + np.exp(-x)))


@given(small_floats)
def test_relu(a):
    if a > 0:
        assert operators.relu(a) == a
    else:
        assert operators.relu(a) == 0.0


## property testing: testing math properties for correctness


@given(small_floats, small_floats)
def test_symmetric(x, y):
    assert operators.mul(x, y) == operators.mul(y, x)
    assert operators.add(x, y) == operators.add(y, x)


@given(small_floats, small_floats, small_floats)
def test_distribute(x, y, z):
    assert_close(
        operators.mul(z, operators.add(x, y)),
        operators.add(operators.mul(z, x), operators.mul(z, y)),
    )


@given(small_floats, small_floats, small_floats, small_floats)
def test_zip_with(a, b, c, d):
    assert_close(operators.addLists([a, b], [c, d]), [a + c, b + d])


@given(
    lists(small_floats, min_size=5, max_size=5),
    lists(small_floats, min_size=5, max_size=5),
)
def test_property(ls1, ls2):
    a = operators.sum(ls1) + operators.sum(ls2)
    b = operators.sum(operators.addLists(ls1, ls2))
    assert_close(a, b)


@given(lists(small_floats))
def test_sum(ls):
    assert_close(operators.sum(ls), sum(ls))


@given(small_floats, small_floats, small_floats)
def test_prod(x, y, z):
    assert_close(operators.prod([x, y, z]), x * y * z)
