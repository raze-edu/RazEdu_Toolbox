import unittest
import math
import sys
import os

# Ensure the Cyron module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Cyron import Cyphon
except ImportError:
    print("Failed to import Cyron.Cyphon. Make sure the module is built.")
    sys.exit(1)

class TestCyronCyphon(unittest.TestCase):

    def test_dot_product(self):
        inputs = [1.0, 2.0, 3.0]
        weights = [0.5, 0.5, 0.5]
        expected = 1.0*0.5 + 2.0*0.5 + 3.0*0.5
        result = Cyphon.dot_product(inputs, weights)
        self.assertAlmostEqual(result, expected)

    def test_dot_product_mismatch(self):
        inputs = [1.0, 2.0]
        weights = [0.5, 0.5, 0.5]
        with self.assertRaises(ValueError):
            Cyphon.dot_product(inputs, weights)

    def test_sigmoid(self):
        self.assertAlmostEqual(Cyphon.sigmoid(0), 0.5)
        self.assertAlmostEqual(Cyphon.sigmoid(710), 1.0) # Overflow check
        self.assertAlmostEqual(Cyphon.sigmoid(-710), 0.0) # Underflow check

    def test_relu(self):
        self.assertEqual(Cyphon.relu(10), 10)
        self.assertEqual(Cyphon.relu(-10), 0)
        self.assertEqual(Cyphon.relu(0), 0)

    def test_tanh(self):
        self.assertAlmostEqual(Cyphon.tanh(0), 0.0)
        self.assertAlmostEqual(Cyphon.tanh(10), math.tanh(10))
        self.assertAlmostEqual(Cyphon.tanh(-10), math.tanh(-10))

    def test_softmax(self):
        inputs = [1.0, 2.0, 3.0]
        result = Cyphon.softmax(inputs)
        total = sum(result)
        self.assertAlmostEqual(total, 1.0)
        
        # Check relative values (3.0 should have highest probability)
        self.assertTrue(result[2] > result[1] > result[0])

if __name__ == '__main__':
    unittest.main()
