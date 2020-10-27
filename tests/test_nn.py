import jtorch
from hypothesis import given
from .strategies import tensors, assert_close
import pytest


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t):
    out = jtorch.avgpool2d(t, (2, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = jtorch.avgpool2d(t, (2, 1))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = jtorch.avgpool2d(t, (1, 2))
    assert (
        out[0, 0, 0, 0]
        == sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    jtorch.grad_check(lambda t: jtorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_2
@given(tensors(shape=(1, 1, 4, 4)))
def test_max(t):
    out = jtorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = jtorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = jtorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t):
    q = jtorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = jtorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0


@pytest.mark.task4_1
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t):
    q = jtorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = jtorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    jtorch.grad_check(lambda a: jtorch.softmax(a, dim=2), t)


@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 6, 6)), tensors(shape=(1, 1, 2, 3)))
def test_conv(input, weight):
    jtorch.grad_check(jtorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_3
@given(tensors(shape=(2, 1, 6, 6)), tensors(shape=(1, 1, 2, 3)))
def test_conv_batch(input, weight):
    jtorch.grad_check(jtorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_3
@given(tensors(shape=(2, 2, 6, 6)), tensors(shape=(3, 2, 2, 3)))
def test_conv_channel(input, weight):
    jtorch.grad_check(jtorch.Conv2dFun.apply, input, weight)


@pytest.mark.task4_3
def test_conv2():
    t = jtorch.tensor_fromlist(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]
    ).view(1, 1, 4, 4)
    t.requires_grad_(True)

    t2 = jtorch.tensor_fromlist([[1, 1], [1, 1]]).view(1, 1, 2, 2)
    t2.requires_grad_(True)
    out = jtorch.Conv2dFun.apply(t, t2)
    out.sum().backward()

    jtorch.grad_check(jtorch.Conv2dFun.apply, t, t2)
