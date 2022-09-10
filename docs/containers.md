
class Module
  class Parameter -> Designed to hold tensor, but can hold any value for testing.


class Variable
  def chain_rule
  class VariableWithDeriv
  class History
    class Context
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

