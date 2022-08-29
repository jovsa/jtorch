# Operators

Contains implementation of all operators. These are used as building blocks units elsewhere.

* [Code location](https://github.com/jovsa/jtorch/blob/main/jtorch/operators.py)
* [Unit tests](https://github.com/jovsa/jtorch/blob/main/tests/tests_operators.py)

## Mathematical Operators

* mul:
  * `f(x, y) = x * y`
* id
  * `f(x) = x`
* add
  * `f(x, y) = x + y`
* neg
  * `f(x) = -x`
* lt
  * `f(x) = 1.0 if x is less than y else 0.0`
* eq
  * `f(x) = 1.0 if x is equal to y else 0.0`
* max
  * `f(x) = x if x is greater than y else y`
* sigmoid
  * `f(x) =  1.0/(1.0 + e^(-x))`
  * For stability, when x < 0, `f(x) =  e^(x)/(1.0 + e^(x))`
* relu
  * `f(x) = x if x is greater than 0, else 0`
* relu_back
  * `f(x, c) = c if x is greater than 0 else 0`
* log
  * `f(x) = log(x)`
  * This is the natural logarithm
* log_back
  * `f(x, c) = c / (x + epsilon)`
  * Formulated as `c * dy/dx{log(x)}`
    * Since log is the natural log, `dy/dx{y = ln(x))} = 1/x`
    * For `dy/dx{y = log_a{x}} = 1/(x*ln(a))`
* inv
  * `f(x) = 1/x`
* inv_back
  * `f(x, c) = -(1.0/x^2)*c`
  * Formulated as `c * dy/dx{y=inv(x)}`
    * Since the `inv(x) = x^-1`, then `dy/dx{y=x^-1} = -1(x^-2)`
* exp
  * `f(x) = e^x`
  * This operator doesn't have a 'backward' operator since the derivative of `dy/dx{y = exp(x)} = exp(x)`


## Listwise Operators

The following operators are written to operate on lists. This formulation extends nicely when
we want to apply them to tensors.

* map
  * Takes as list and applies a `fn` to each element. Returns a new list.
  * wikipedia [page](https://en.wikipedia.org/wiki/Map_(higher-order_function))
* reduce
  * Takes a list of elements. Cumulatively applies a `fn` to each element. Return a single value.
* zipWith
  * Takes two equally sized lists. Produces a new list by applying a `fn` to each pair of elements.
  * wikipedia [page](https://en.wikipedia.org/wiki/Map_(higher-order_function))


## Composite Operators

These operators use the mathematical and listwise operators as building blocks.
These are mainly used to tests the listwise operators.

* negList
  * Use the `map` to apply `neg` to each element in a list.
* addLists
  * Takes two equally sized lists. Does an element wise addition for each element and produces a new list.
* sum
  * Takes a list of elements. `reduce` the list with the `add` operator.
* prod
  * Takes a list of elements. `reduce` the list with the `mul` operator.
