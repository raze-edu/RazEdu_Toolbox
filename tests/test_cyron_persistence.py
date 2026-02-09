import unittest
import sys
import os
import json
import array

# Ensure the Cyron module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Cyron import NeuralNetwork
except ImportError:
    from Cyron.Network import NeuralNetwork

class TestCyronPersistence(unittest.TestCase):
    
    def setUp(self):
        self.filename = 'test_weights.json'
        if os.path.exists(self.filename):
            os.remove(self.filename)
            
        self.nn = NeuralNetwork()
        self.nn.add_layer(2, 2, activation='relu')

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_save_new(self):
        self.nn.save_weights(self.filename)
        self.assertTrue(os.path.exists(self.filename))
        
        with open(self.filename, 'r') as f:
            data = json.load(f)
            
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['input_size'], 2)
        # History should have 1 item (start values)
        self.assertEqual(len(data[0]['weights_history'][0]), 1) 

    def test_load_weights(self):
        # Save original weights
        original_weights = [list(w) for w in self.nn.layers[0]['weights']]
        self.nn.save_weights(self.filename)
        
        # Reset network
        new_nn = NeuralNetwork()
        new_nn.load_weights(self.filename)
        
        loaded_weights = [list(w) for w in new_nn.layers[0]['weights']]
        
        self.assertEqual(original_weights, loaded_weights)

    def test_save_delta_and_version(self):
        # 1. Save Initial (Version 0)
        self.nn.save_weights(self.filename)
        w0 = [list(w) for w in self.nn.layers[0]['weights']]
        
        # 2. Modify Weights
        # Add 1.0 to all weights
        for node_weights in self.nn.layers[0]['weights']:
            for i in range(len(node_weights)):
                node_weights[i] += 1.0
        
        w1 = [list(w) for w in self.nn.layers[0]['weights']]
        self.assertNotEqual(w0, w1)
        
        # 3. Save Update (Version 1)
        self.nn.save_weights(self.filename)
        
        # Verify JSON structure
        with open(self.filename, 'r') as f:
            data = json.load(f)
            # Should have 2 items in history: [start, delta]
            self.assertEqual(len(data[0]['weights_history'][0]), 2)
            # Delta should be roughly 1.0
            delta = data[0]['weights_history'][0][1]
            self.assertAlmostEqual(delta[0], 1.0)

        # 4. Load Version 0 (Start)
        nn_v0 = NeuralNetwork()
        nn_v0.load_weights(self.filename, version=0)
        w_v0 = [list(w) for w in nn_v0.layers[0]['weights']]
        
        # Version 0 should match original w0
        for i in range(len(w0)):
            for j in range(len(w0[i])):
                self.assertAlmostEqual(w_v0[i][j], w0[i][j])
                
        # 5. Load Version 1 (Latest)
        nn_v1 = NeuralNetwork()
        nn_v1.load_weights(self.filename, version=1) # or None
        w_v1 = [list(w) for w in nn_v1.layers[0]['weights']]
        
        # Version 1 should match modified w1
        for i in range(len(w1)):
            for j in range(len(w1[i])):
                self.assertAlmostEqual(w_v1[i][j], w1[i][j])

if __name__ == '__main__':
    unittest.main()
