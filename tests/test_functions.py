import jtorch
import pytest
from .strategies import assert_close


@pytest.mark.task3_1
def test_mm():
    a = jtorch.rand((2, 3))
    b = jtorch.rand((3, 4))
    c = jtorch.matmul(a, b)

    c2 = (a.view(2, 3, 1) * b.view(1, 3, 4)).sum(1).view(2, 4)

    print(c)
    print(c2)
    for ind in c._tensor.indices():
        assert_close(c[ind], c2[ind])


@pytest.mark.task3_1
def test_broad_mm():
    a = jtorch.rand((2, 2, 3))
    b = jtorch.rand((2, 3, 4))
    c = jtorch.matmul(a, b)

    c2 = (a.view(2, 2, 3, 1) * b.view(2, 1, 3, 4)).sum(2).view(2, 2, 4)

    print(c)
    print(c2)
    for ind in c._tensor.indices():
        assert_close(c[ind], c2[ind])


# @pytest.mark.task3_4
# def test_cuda_mm():
#     a = jtorch.rand((2, 2, 3))
#     b = jtorch.rand((2, 3, 4))
#     c = jtorch.cuda_matmul(a, b)

#     c2 = (a.view(2, 2, 3, 1) * b.view(2, 1, 3, 4)).sum(2).view(2, 2, 4)

#     print(c)
#     print(c2)
#     for ind in c._tensor.indices():
#         assert_close(c[ind], c2[ind])
