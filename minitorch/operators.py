"""Collection of the core mathematical operators used throughout the code base."""

from typing import Callable, Iterable
import math


# ## Task 0.1

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$

# TODO: Implement for Task 0.1.


def mul(a: float, b: float) -> float:
    """Multiple a with b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        float: result of a * b

    """
    return a * b


def id(a: float) -> float:
    """Return the original number of input

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: a

    """
    return a


def add(a: float, b: float) -> float:
    """Add a with b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        float: result of a + b

    """
    return a + b


def neg(a: float) -> float:
    """Return negative of a

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: -a

    """
    return -1.0 * a


def lt(a: float, b: float) -> bool:
    """Check if a is less than b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        bool: a < b

    """
    return a < b


def eq(a: float, b: float) -> bool:
    """Check if a equals to b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        bool: a == b

    """
    return a == b


def max(a: float, b: float) -> float:
    """Return the max number between a and b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        float: the larger one between a and b

    """
    return a if a > b else b


def is_close(a: float, b: float) -> bool:
    """Return if a is close to b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        bool: |a - b| < eps

    """
    return abs(a - b) < 1e-4


def sigmoid(a: float) -> float:
    """Return sigmoid(a)

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: 1 / (1 + e(-a))

    """
    return 1 / (1 + math.exp(-a))


def relu(a: float) -> float:
    """Return relu(a)

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: max(0, a)

    """
    return max(a, 0.0)


def log(a: float) -> float:
    """Return log(a)

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: log(a)

    """
    return math.log(a)


def exp(a: float) -> float:
    """Return exp(a)

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: exp(a)

    """
    return math.exp(a)


def log_back(a: float, b: float) -> float:
    """Return gradient of log(a), and then multiple b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        float: b * gradient(log(a))

    """
    return b / a


def inv(a: float) -> float:
    """Inverse of a

    Args:
    ----
        a (float): input number

    Returns:
    -------
        float: a^(-1)

    """
    return 1 / a


def inv_back(a: float, b: float) -> float:
    """Return gradient of inv(a), and then multiple b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        float: b * gradient(1/a)

    """
    return -b / (a * a)


def relu_back(a: float, b: float) -> float:
    """Return gradient of relu(a), and then multiple b

    Args:
    ----
        a (float): first number
        b (float): second number

    Returns:
    -------
        float: b * gradient(relu(a))

    """
    return 0 if a < 0 else b


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists

# TODO: Implement for Task 0.3.


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
         A function that takes a list, applies `fn` to each element, and returns a
         new list

    """

    # TODO: Implement for Task 0.3.
    def res(x: Iterable[float]) -> Iterable[float]:
        return [fn(i) for i in x]

    return res


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Use `map` and `neg` to negate each element in `ls`"""
    # TODO: Implement for Task 0.3.
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipwith (or map2).

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: combine two values

    Returns:
    -------
         Function that takes two equally sized lists `ls1` and `ls2`, produce a new list by
         applying fn(x, y) on each pair of elements.

    """

    # TODO: Implement for Task 0.3.
    def res(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
        aa = [i for i in a]
        bb = [i for i in b]
        return [fn(aa[i], bb[i]) for i in range(len(aa))]

    return res


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add the elements of `ls1` and `ls2` using `zipWith` and `add`"""
    # TODO: Implement for Task 0.3.
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
         Function that takes a list `ls` of elements
         $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
         fn(x_1, x_0)))`

    """

    # TODO: Implement for Task 0.3.
    def res(data: Iterable[float]) -> float:
        last = start
        for i in data:
            last = fn(i, last)
        return last

    return res


def sum(ls: Iterable[float]) -> float:
    """Sum up a list using `reduce` and `add`."""
    # TODO: Implement for Task 0.3.
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Product of a list using `reduce` and `mul`."""
    # TODO: Implement for Task 0.3.
    return reduce(mul, 1.0)(ls)
