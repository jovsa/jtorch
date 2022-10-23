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

* A multi-dimentional array of arbitrary length. This is the main container for storing data. It is a subclass of `Variable`.
* Designed to be use native python lists as the underlying data structure.
* Assumes the data passed in is of type `TensorData`.

### TensorData
* A wrapper around a python list. This is the underlying data structure for `Tensor`.
* Given data, calculates the strides. Strides is a tuple that provides the mapping from user indexing to the position in the 1-D storage
* Given data, calculates the shapes. Shapes is a tuple that provides the shape of the tensor.


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
