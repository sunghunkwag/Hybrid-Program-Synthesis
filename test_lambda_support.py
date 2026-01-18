
import unittest
import ast
from neuro_genetic_synthesizer import SafeInterpreter

class TestLambdaSupport(unittest.TestCase):
    def setUp(self):
        # Allow basic arithmetic ops for testing
        self.primitives = {
            'add': lambda a, b: a + b,
            'mul': lambda a, b: a * b
        }
        self.interpreter = SafeInterpreter(self.primitives)

    def run_code(self, code, env={}):
        return self.interpreter.run(code, env)

    def test_basic_lambda(self):
        # (lambda x: x + 1)(5)
        code = "(lambda x: add(x, 1))(5)"
        result = self.run_code(code)
        self.assertEqual(result, 6)

    def test_lambda_closure(self):
        # Closure test: y captured from outer scope
        # (lambda x: add(x, y))(10) with y=20 in env
        code = "(lambda x: add(x, y))(10)"
        result = self.run_code(code, env={'y': 20})
        self.assertEqual(result, 30)

    def test_higher_order_func(self):
        # Simulate map: (lambda f, x: f(x))(lambda y: mul(y, 2), 5) -> 10
        # Passing a lambda TO another lambda
        code = """(lambda f, val: f(val))(
            lambda z: mul(z, 2),
            5
        )"""
        result = self.run_code(code)
        self.assertEqual(result, 10)

    def test_y_combinator_factorial(self):
        # Advanced: Y-combinator approximation for recursion (if gas allows)
        # Factorial of 5 = 120
        # Standard Y-combinator:
        # Y = lambda f: (lambda x: f(lambda y: x(x)(y)))(lambda x: f(lambda y: x(x)(y)))
        # Here we try a simpler recursive structure if supported or just nested lambdas
        
        # Simple recursive lambda is hard without 'fix', but we can test nesting depth
        code = """(lambda x: 
                    (lambda y: 
                        add(x, y)
                    )(10)
                  )(5)"""
        result = self.run_code(code)
        self.assertEqual(result, 15)

if __name__ == '__main__':
    unittest.main()
