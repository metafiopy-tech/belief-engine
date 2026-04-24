from typing import Union

class FibonacciError(Exception):
    """Custom exception for Fibonacci function errors."""
    pass

def fibonacci(n: int) -> int:
    """
    Returns the Nth Fibonacci number using iteration.

    :param n: The position in the Fibonacci sequence (0-based index).
    :raises FibonacciError: If n is negative.
    :return: The Nth Fibonacci number.
    """
    if n < 0:
        raise FibonacciError("The input must be a non-negative integer.")
    
    if n == 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b