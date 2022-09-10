import jtorch
import pytest
from jtorch import History, Variable


class Temp(jtorch.FunctionBase):
    "Implements additions"

    @staticmethod
    def backward(ctx, d_output):
        return d_output, d_output


class Temp2(jtorch.FunctionBase):
    "Implements additions"

    @staticmethod
    def backward(ctx, d_output):
        x = ctx.saved_values
        return d_output, x * d_output


def test_chain_rule():

    # case: only constants through chain rule.
    constant = jtorch.Variable(None)
    for variable_with_deriv in Temp.chain_rule(
        ctx=None, inputs=[constant, constant], d_output=5
    ):
        assert False

    # case: constant and variables mix
    var = jtorch.Variable(History())
    constant = jtorch.Variable(None)
    for variable_with_deriv in Temp.chain_rule(
        ctx=None, inputs=[var, constant], d_output=5
    ):
        assert variable_with_deriv.variable.name == var.name
        assert variable_with_deriv.deriv == 5

    # case: save_for_backward is correctly used
    ctx = jtorch.Context()
    ctx.save_for_backward(10)
    for variable_with_deriv in Temp2.chain_rule(
        ctx=ctx, inputs=[constant, var], d_output=5
    ):
        assert variable_with_deriv.variable.name == var.name
        assert variable_with_deriv.deriv == 5 * 10

    # case: right output from backwards is used
    ctx = jtorch.Context()
    ctx.save_for_backward(10)
    for variable_with_deriv in Temp2.chain_rule(
        ctx=ctx, inputs=[var, constant], d_output=5
    ):
        assert variable_with_deriv.variable.name == var.name
        assert variable_with_deriv.deriv == 5


def test_backprop():

    var = Variable(History())
    var2 = Variable(History(Temp, None, [0, var]))
    var2.backward(5)
    assert var.derivative == 5

    var = Variable(History())
    var2 = Variable(History(Temp, None, [0, var]))
    var3 = Variable(History(Temp, None, [0, var2]))
    var3.backward(5)
    assert var.derivative == 5

    var1 = Variable(History())
    var2 = Variable(History(Temp, None, [0, var1]))
    var3 = Variable(History(Temp, None, [0, var1]))
    var4 = Variable(History(Temp, None, [var2, var3]))
    var4.backward(5)
    assert var1.derivative == 10

    var0 = Variable(History())
    var1 = Variable(History(Temp, None, [0, var0]))
    var2 = Variable(History(Temp, None, [0, var1]))
    var3 = Variable(History(Temp, None, [0, var1]))
    var4 = Variable(History(Temp, None, [var2, var3]))
    var4.backward(5)
    assert var0.derivative == 10
