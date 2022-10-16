# Containers

This describes all objects that are used to store and manipulate data.

## Variable
You can find the definition in `class Variable`. This is a the base container for storing any data or compute operations that you want to backpropagation on.

Some usefull abstractions that this container provides:
* `class FunctionBase`
* `class History`:
* `class VariableWithDeriv`
* `class Context`

Also some usefull functions:
* `def chain_rule`
* `def is_leaf`
* `def is_constant`
* `def backpropagate`


## Tensor

```
Tensor <- Variable
  Function
    FunctionBase
    Scalar <- Variable
      ScalarFunction
        FunctionBase
```


## Module
```
class Module
  class Parameter -> Designed to hold tensor, but can hold any value for testing.
```
