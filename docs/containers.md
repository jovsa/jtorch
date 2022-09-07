
class Module
  class Parameter -> Designed to hold tensor, but can hold any value for testing.


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

