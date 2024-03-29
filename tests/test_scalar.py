import jtorch
from hypothesis import given
from .strategies import scalars, assert_close
import pytest


# @pytest.mark.task1_1
def test_central_diff():
    d = jtorch.central_difference(jtorch.operators.id, 5, arg=0)
    assert_close(d, 1.0)
    d = jtorch.central_difference(jtorch.operators.add, 5, 10, arg=0)
    assert_close(d, 1.0)

    d = jtorch.central_difference(jtorch.operators.mul, 5, 10, arg=0)
    assert_close(d, 10.0)
    d = jtorch.central_difference(jtorch.operators.mul, 5, 10, arg=1)
    assert_close(d, 5.0)


one_arg = [
    ("neg", lambda a: -a),
    ("addconstant", lambda a: 5 + a),
    ("subconstant", lambda a: a - 5),
    ("mult", lambda a: 5 * a),
    ("div", lambda a: a / 5),
    ("sig", lambda a: a.sigmoid(), lambda a: jtorch.operators.sigmoid(a)),
    (
        "log",
        lambda a: (a + 100000).log(),
        lambda a: jtorch.operators.log(a + 100000),
    ),
    (
        "exp",
        lambda a: (a - 100000).exp(),
        lambda a: jtorch.operators.exp(a - 100000),
    ),
    ("relu", lambda a: (a + 5.5).relu(), lambda a: jtorch.operators.relu(a + 5.5)),
]


@given(scalars(min_value=-100, max_value=100))
@pytest.mark.task1_2
@pytest.mark.parametrize("fn", one_arg)
def test_one_args(fn, t1):
    check = 1
    if len(fn) > 2:
        check = 2
    assert_close(fn[1](t1).data, fn[check](t1.data))


@given(scalars(min_value=-100, max_value=100))
@pytest.mark.task1_4
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(fn, t1):
    jtorch.derivative_check(fn[1], t1)


two_arg = [
    ("add", lambda a, b: a + b),
    ("gt", lambda a, b: a + 1.2 > b),
    ("lt", lambda a, b: a + 1.2 < b),
    ("mul", lambda a, b: a * b),
    ("div", lambda a, b: a / (b + 5.5)),
]


@given(scalars(min_value=-100, max_value=100), scalars(min_value=-100, max_value=100))
@pytest.mark.task1_4
@pytest.mark.parametrize("fn", two_arg)
def test_two_derivative(fn, t1, t2):
    jtorch.derivative_check(fn[1], t1, t2)


def test_scalar_name():
    x = jtorch.Scalar(10, name="x")
    y = (x + 10.0) * 20
    y.name = "y"
    hash(y)
    return y
