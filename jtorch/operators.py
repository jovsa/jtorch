import math
from typing import Callable, List

EPSILON = 1e-6


def mul(x: float, y: float) -> float:
    ":math:`f(x, y) = x * y`"
    return x * y


def id(x: float) -> float:
    ":math:`f(x) = x`"
    return x


def add(x: float, y: float) -> float:
    ":math:`f(x, y) = x + y`"
    return x + y


def neg(x: float) -> float:
    ":math:`f(x) = -x`"
    return -x


def lt(x: float, y: float) -> float:
    ":math:`f(x) = 1.0 if x is less than y else 0.0"
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    ":math:`f(x) = 1.0 if x is equal to y else 0.0"
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    ":math:`f(x) = x if x is greater than y else y"
    return x if x > y else y


def sigmoid(x: float) -> float:
    ":math:`f(x) =  \frac{1.0}{(1.0 + e^{-x})}`"
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    ":math:`f(x) = x if x is greater than 0, else 0"
    return x if x > 0 else 0.0


def relu_back(x: float, y: float) -> float:
    ":math:`f(x, y) =` y if x is greater than 0 else 0"
    return y if x > 0 else 0.0


def log(x: float) -> float:
    ":math:`f(x) = log(x)`"
    return math.log(x + EPSILON)


def exp(x: float) -> float:
    ":math:`f(x) = e^{x}`"
    return math.exp(x)


def log_back(a: float, b: float) -> float:
    return b / (a + EPSILON)


def inv(x: float) -> float:
    ":math:`f(x) = 1/x`"
    return 1.0 / x


def inv_back(a: float, b: float) -> float:
    return -(1.0 / a**2) * b


def map(fn) -> Callable:
    """
    Higher-order map.

    Args:
        fn (one-arg function): Function from one value to one value.

    Returns:
        function : A function that takes a list, applies `fn` to each element, and returns a
        new list
    """

    def _map(ls):
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def negList(ls: List) -> List:
    "Use :func:`map` and :func:`neg` to negate each element in `ls`"
    return map(neg)(ls)


def zipWith(fn) -> Callable:
    """
    Higher-order zipwith.

    Args:
        fn (two-arg function): combine two values

    Returns:
        function : takes two equally sized lists `ls1` and `ls2`, produce a new list by
        applying fn(x, y) one each pair of elements.

    """

    def _zipWith(ls1, ls2):
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def addLists(ls1: List, ls2: List) -> List:
    "Add the elements of `ls1` and `ls2` using :func:`zipWith` and :func:`add`"
    return zipWith(add)(ls1, ls2)


def reduce(fn: Callable, start: float) -> Callable:
    r"""
    Higher-order reduce.

    Args:
        fn (two-arg function): combine two values
        start (float): start value :math:`x_0`

    Returns:
        function : function that takes a list `ls` of elements
        :math:`x_1 \ldots x_n` and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """

    def _reduce(ls):
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def sum(ls: List) -> List:
    """
    Sum up a list using :func:`reduce` and :func:`add`.
    """
    return reduce(add, 0.0)(ls)


def prod(ls: List) -> List:
    """
    Product of a list using :func:`reduce` and :func:`mul`.
    """
    return reduce(mul, 1.0)(ls)
