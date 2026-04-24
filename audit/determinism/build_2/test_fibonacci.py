import unittest

def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("Input should be a non-negative integer.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

class TestFibonacci(unittest.TestCase):
    
    def test_fibonacci_zero(self):
        self.assertEqual(fibonacci(0), 0)
    
    def test_fibonacci_one(self):
        self.assertEqual(fibonacci(1), 1)
    
    def test_fibonacci_two(self):
        self.assertEqual(fibonacci(2), 1)
    
    def test_fibonacci_three(self):
        self.assertEqual(fibonacci(3), 2)
    
    def test_fibonacci_ten(self):
        self.assertEqual(fibonacci(10), 55)
    
    def test_fibonacci_negative(self):
        with self.assertRaises(ValueError):
            fibonacci(-1)

if __name__ == '__main__':
    unittest.main()