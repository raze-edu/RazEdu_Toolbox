import unittest
import sys
import os
import time

# Ensure the Cyron module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Cyron import NeuralNetwork
except ImportError:
    # If standard import fails, try direct import from file location if needed, 
    # but sys.path above should handle it.
    from Cyron.Network import NeuralNetwork

class TestCyronNetwork(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork()
        # Input layer: 3 inputs -> 4 hidden nodes
        self.nn.add_layer(3, 4, activation='relu')
        # Output layer: 4 hidden -> 2 outputs
        self.nn.add_layer(4, 2, activation='softmax')

    def test_forward_shape(self):
        inputs = [0.1, 0.2, 0.3]
        output = self.nn.forward(inputs)
        self.assertEqual(len(output), 2)
        # Softmax sum should be approx 1.0
        self.assertAlmostEqual(sum(output), 1.0, places=5)

    def test_threaded_processing(self):
        batch_size = 100
        inputs = [[0.1, 0.2, 0.3] for _ in range(batch_size)]
        
        # Sequential processing (manual)
        start_seq = time.time()
        results_seq = [self.nn.forward(inp) for inp in inputs]
        end_seq = time.time()
        
        # Threaded processing
        start_thread = time.time()
        results_thread = self.nn.process_batch_threaded(inputs, max_workers=4)
        end_thread = time.time()
        
        self.assertEqual(len(results_thread), batch_size)
        
        # Results should be identical
        for r_seq, r_thread in zip(results_seq, results_thread):
            for v1, v2 in zip(r_seq, r_thread):
                self.assertAlmostEqual(v1, v2)
                
        print(f"\nSequential Time: {end_seq - start_seq:.5f}s")
        print(f"Threaded Time:   {end_thread - start_thread:.5f}s")

if __name__ == '__main__':
    unittest.main()
