import random
import array
import json
import os
from concurrent.futures import ThreadPoolExecutor
try:
    from . import Cyphon
except ImportError:
    import Cyphon

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation='sigmoid'):
        """
        Adds a fully connected layer to the network.
        Weights are initialized randomly between -1 and 1.
        """
        weights = []
        # Create a matrix of weights: output_size x input_size
        # Flattened list for simplicity in this example, or list of lists
        # Cyphon.dot_product takes 1D arrays, so we store weights as a list of 1D arrays (one per output node)
        for _ in range(output_size):
            node_weights = array.array('d', [random.uniform(-1, 1) for _ in range(input_size)])
            weights.append(node_weights)
            
        layer = {
            'weights': weights,
            'activation': activation,
            'input_size': input_size,
            'output_size': output_size
        }
        self.layers.append(layer)

    def _activate(self, value, activation_type):
        if activation_type == 'sigmoid':
            return Cyphon.sigmoid(value)
        elif activation_type == 'relu':
            return Cyphon.relu(value)
        elif activation_type == 'tanh':
            return Cyphon.tanh(value)
        return value

    def forward(self, inputs):
        """
        Performs a forward pass for a single input vector.
        Input must be an iterable (converted to array.array('d') internally).
        """
        current_input = array.array('d', inputs)
        
        for layer in self.layers:
            next_input = array.array('d')
            weights = layer['weights']
            activation = layer['activation']
            
            # For each node in this layer
            if activation == 'softmax':
                # Softmax puts all outputs together
                raw_outputs = array.array('d')
                for node_weights in weights:
                    val = Cyphon.dot_product(current_input, node_weights)
                    raw_outputs.append(val)
                # Softmax returns a list
                current_input = array.array('d', Cyphon.softmax(raw_outputs))
            else:
                for node_weights in weights:
                    val = Cyphon.dot_product(current_input, node_weights)
                    activated_val = self._activate(val, activation)
                    next_input.append(activated_val)
                current_input = next_input
                
        return list(current_input)

    def process_batch_threaded(self, batch_inputs, max_workers=4):
        """
        Processes a batch of inputs in parallel using multithreading.
        Since Cyphon releases the GIL for heavy ops, this provides speedup.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map preserves order
            results = list(executor.map(self.forward, batch_inputs))
        return results

    def save_weights(self, filepath):
        """
        Saves weights to a JSON file.
        If file exists, appends the delta of current weights vs last saved weights.
        If file does not exist, saves current weights as start values.
        """
        existing_data = []
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                pass # Treat as new file if corrupt/empty
        
        new_data = []
        
        for i, layer in enumerate(self.layers):
            layer_history = []
            
            # If we have existing data for this layer, we need to reconstruct the last state
            # to calculate deltas.
            existing_layer_data = existing_data[i] if i < len(existing_data) else None
            
            curr_weights_list = [list(w) for w in layer['weights']]
            
            if existing_layer_data:
                # Structure: {..., "weights_history": [ [ [w0...], [d1...], ... ], ... ] }
                histories = existing_layer_data.get('weights_history', [])
                
                new_layer_history = []
                for node_idx, curr_node_weights in enumerate(curr_weights_list):
                    if node_idx < len(histories):
                        node_history = histories[node_idx]
                        # Reconstruct last weights
                        last_weights = list(node_history[0]) # Start with initial
                        for delta in node_history[1:]:
                            for w_idx in range(len(last_weights)):
                                last_weights[w_idx] += delta[w_idx]
                        
                        # Calculate delta
                        current_delta = [c - l for c, l in zip(curr_node_weights, last_weights)]
                        
                        # Append delta to history
                        node_history.append(current_delta)
                        new_layer_history.append(node_history)
                    else:
                        # New node? Shouldn't happen if architecture is static, but handle as new start
                        new_layer_history.append([curr_node_weights])
                
                layer_entry = existing_layer_data
                layer_entry['weights_history'] = new_layer_history
            else:
                # New layer or new file
                # Save just the start values
                weights_history = [[node_w] for node_w in curr_weights_list]
                layer_entry = {
                    'activation': layer['activation'],
                    'input_size': layer['input_size'],
                    'output_size': layer['output_size'],
                    'weights_history': weights_history
                }
            
            new_data.append(layer_entry)
            
        with open(filepath, 'w') as f:
            json.dump(new_data, f, indent=2)

    def load_weights(self, filepath, version=None):
        """
        Loads weights from a JSON file.
        version: Integer specifying which delta step to sum up to. 
                 None means latest (apply all deltas).
                 0 means just the start values.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weights file not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.layers = []
        for layer_data in data:
            activation = layer_data['activation']
            input_size = layer_data['input_size']
            output_size = layer_data['output_size']
            weights_history = layer_data['weights_history']
            
            reconstructed_weights = []
            
            for node_history in weights_history:
                # Start with initial values
                current_node_weights = list(node_history[0])
                
                # Determine how many deltas to apply
                deltas = node_history[1:]
                if version is not None:
                    # If version is 0, we apply 0 deltas.
                    # If version is 1, we apply 1 delta (index 0).
                    # Check bounds
                    idx_limit = version 
                    if idx_limit > len(deltas):
                        idx_limit = len(deltas)
                    deltas = deltas[:idx_limit]
                
                # Apply deltas
                for delta in deltas:
                    for w_idx in range(len(current_node_weights)):
                        current_node_weights[w_idx] += delta[w_idx]
                
                reconstructed_weights.append(array.array('d', current_node_weights))
                
            self.layers.append({
                'weights': reconstructed_weights,
                'activation': activation,
                'input_size': input_size,
                'output_size': output_size
            })
