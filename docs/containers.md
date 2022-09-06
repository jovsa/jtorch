
class Module

class Variable
  class Context
  class History
  class VariableWithDeriv
  class FunctionBase
  def is_leaf
  def is_constant
  def backpropagate


Tensor <- Variable
  Function
    FunctionBase
    Scalar <- Variable
      ScalarFunction
        FunctionBase

