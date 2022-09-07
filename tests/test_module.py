import jtorch
import pytest

VAL = 40
VAL_A = 50
VAL_B = 100
MODULE_2_EXTRA_PARAMS_1 = 5
MODULE_2_EXTRA_PARAMS_2 = 10
PARAM_VAL = 1123


class Module1(jtorch.Module):
    def __init__(self):
        super().__init__()
        self.module_a = Module2(MODULE_2_EXTRA_PARAMS_1)
        self.module_b = Module2(MODULE_2_EXTRA_PARAMS_2)
        self.parameter_a = jtorch.Parameter(VAL)


class Module2(jtorch.Module):
    def __init__(self, extra=0):
        super().__init__()
        self.parameter_a = jtorch.Parameter(VAL_A)
        self.parameter_b = jtorch.Parameter(VAL_B)
        self.non_parameter = 10
        for i in range(extra):
            self.add_parameter(f"extra_parameter_{i}", PARAM_VAL)


def test_module():
    "Check the properties of a single module"
    module = Module2()
    try:
        module()
    except Exception as e:
        assert isinstance(e, NotImplementedError) == True

    module.eval()
    assert module.mode == "eval"
    module.train()
    assert module.mode == "train"
    assert len(module.parameters()) == 2

    module = Module2(MODULE_2_EXTRA_PARAMS_2)
    assert len(module.parameters()) == MODULE_2_EXTRA_PARAMS_2 + 2

    module = Module2(MODULE_2_EXTRA_PARAMS_1)
    named_parameters = module.named_parameters()
    assert named_parameters["parameter_a"].value == VAL_A
    assert named_parameters["parameter_b"].value == VAL_B
    assert named_parameters["extra_parameter_0"].value is PARAM_VAL


def test_stacked_module():
    "Check the properties of a stacked module"
    module = Module1()
    module.eval()
    assert module.mode == "eval"
    assert module.module_a.mode == "eval"
    assert module.module_b.mode == "eval"
    module.train()
    assert module.mode == "train"
    assert module.module_a.mode == "train"
    assert module.module_b.mode == "train"

    assert len(module.parameters()) == 1 + 7 + 12

    named_parameters = module.named_parameters()
    assert named_parameters["parameter_a"].value == VAL
    assert named_parameters["module_a.parameter_a"].value == VAL_A
    assert named_parameters["module_a.parameter_b"].value == VAL_B
    assert named_parameters["module_b.parameter_a"].value == VAL_A
    assert named_parameters["module_b.parameter_b"].value == VAL_B
    # from pdb import set_trace; set_trace()

    for i in range(1, MODULE_2_EXTRA_PARAMS_1):
        assert named_parameters[f"module_a.extra_parameter_{i}"].value is PARAM_VAL

    for i in range(1, MODULE_2_EXTRA_PARAMS_2):
        assert named_parameters[f"module_b.extra_parameter_{i}"].value is PARAM_VAL


def test_parameter():
    param = jtorch.Parameter(VAL_A)
    assert param.value == VAL_A
    param.update(VAL_B)
    assert param.value == VAL_B
