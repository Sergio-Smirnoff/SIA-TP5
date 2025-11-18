
import sys
import logging as log
import numpy as np
from tqdm import tqdm
import re
from activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative


class BasicAutoencoder:

    def __init__(
            self,
            # ej: [input_size, l1_hidden_size, l2_hidden_size] 
            # l1 == l2, input_size == output_size, l2 == 2 as middle layer always 2
            architecture=[35, 16, 8, 4, 2], 
            learning_rate=0.001,
            epsilon=1e-4,
            optimizer='sgd',
            adam_b1 = 0.9,
            adam_b2 = 0.999,
            adam_epsilon = 1e-8,
            activation_function='sigmoid',
            seed=42
            ):

        log.info("Initializing Autoencoder...")
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.optimizer = optimizer

       # Additional parameters
        self.seed = seed

        # Errors
        self.error_entropy = 1
        self.error_entropy_ant = np.inf
        self.error_entropy_min = np.inf

        self.error_mse = 1
        self.error_mse_ant = np.inf
        self.error_mse_min = np.inf

        if activation_function == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation_function == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation_function == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative

        # Matrix of weights and biases
        self.wb_initializer(architecture, seed)

        if self.optimizer == 'adam':
            self.initialize_adam(adam_b1, adam_b2, adam_epsilon)

        

        log.info("Autoencoder initialized with architecture: {}".format(architecture))


    def wb_initializer(self, architecture, seed):

        self.seed = seed
        np.random.seed(seed) if seed is not None else None

        architecture = architecture + architecture[-2::-1]  # Complete architecture for encoder + decoder

        self.n_layers = len(architecture)
        self.weights = []
        self.biases = []
        log.info("Initializing weights and biases...")
        for i in range(self.n_layers - 1): 
            # Xavier: W ~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
            # So that the variance of activations is the same across every layer and to avoid exploding/vanishing gradients
            limit = np.sqrt(6 / (architecture[i] + architecture[i+1]))
            W = np.random.uniform(-limit, limit, 
                                 (architecture[i], architecture[i+1]))
            b = np.zeros((1, architecture[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)

        log.info("Weights and biases initialized.")
        

    def initialize_adam(self, beta1, beta2, adam_epsilon):
        """Initialize Adam optimizer momentum and velocity terms."""
        log.info("Initializing Adam optimizer variables...")
        self.m_weights = [np.zeros_like(w) for w in self.weights]
        self.v_weights = [np.zeros_like(w) for w in self.weights]
        self.m_biases = [np.zeros_like(b) for b in self.biases]
        self.v_biases = [np.zeros_like(b) for b in self.biases]
        self.t = 0  # Time step for Adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam_epsilon = adam_epsilon

        log.info("Adam optimizer initialized.")

    def forward(self, X):
        """
        Args:
            X: input dataset
        Returns:
            activations: List of activations per layer
            z_values: List of Z values per layer
        """
        log.info("Performing forward propagation...")
        activations = [X]
        z_values = []
        
        A = X
        #TODO check X state after each pass
        for i in range(self.n_layers - 1):
            log.debug("Layer {}: Input shape A: {}".format(i+1, np.array(A).shape))
            log.debug("Layer {}: Weights shape: {}, Biases shape: {}".format(i+1, self.weights[i].shape, self.biases[i].shape))
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                A = self.activation(Z)
            else:   
                A = sigmoid(Z)  # Output layer with sigmoid

            z_values.append(Z)
            activations.append(A)
            #log.debug("Layer {}: Z: {}, A: {}".format(i+1, Z, A))
        
        log.info("Forward propagation completed.")

        return activations, z_values

    def backward(self, X, Y, activations, z_values):
        """
        Args:
            X: Input data
            Y: Desired output (target)
            activations: List of activations from the forward pass
            z_values: List of Z values from the forward pass
        Returns:
            Gradients (dW, db)
        """
        log.info("Performing backward propagation...")
        m = X.shape[0]

        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        dZ = activations[-1] - Y
        dA = 1

        for i in reversed(range(len(self.weights))):

            if i == len(self.weights) - 1:
                # Output layer: already have dZ
                pass
            else:
                # Hidden layers: use derivative of the hidden activation
                dZ = dA * self.activation_derivative(z_values[i])

            # Compute gradients
            dW[i] = np.dot(activations[i].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m

            # Prepare dA for previous layer
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)

        log.info("Backward propagation completed.")
        return dW, db
    
    def update_parameters(self, dW, db):
        """
        Update weights and biases using gradients with selected optimizer
        Args:
            dW: Gradients for weights
            db: Gradients for biases
        """
        log.info("Updating parameters with {} optimizer...".format(self.optimizer))
        
        if self.optimizer == 'sgd':
            self.update_sgd(dW, db)
        elif self.optimizer == 'adam':
            self.update_adam(dW, db)
        else:
            raise ValueError("Unknown optimizer: {}".format(self.optimizer))

    def update_sgd(self, dW, db):
        """
        Update weights and biases using gradients
        Args:
            dW: Gradients for weights
            db: Gradients for biases
        """
        log.info("Updating parameters...")
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
        log.info("Parameters updated.")

    def update_adam(self, dW, db):
        """Adam optimizer update."""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Update biased first moment estimate (momentum)
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * dW[i]
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * db[i]
            
            # Update biased second raw moment estimate (velocity)
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (dW[i] ** 2)
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (db[i] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_weights_corrected = self.m_weights[i] / (1 - self.beta1 ** self.t)
            m_biases_corrected = self.m_biases[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_weights_corrected = self.v_weights[i] / (1 - self.beta2 ** self.t)
            v_biases_corrected = self.v_biases[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            self.weights[i] -= self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.adam_epsilon)
            self.biases[i] -= self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.adam_epsilon)


    def compute_loss(self, Y_pred, Y_true) -> float:
        """
        Binary Cross-Entropy Loss
        Args:
            Y_pred: Predicted output
            Y_true: True output 
        Returns:
            loss: Computed loss value
        """
        log.info("Computing loss...")
        loss = -np.mean(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8)) 
        # TODO: verificar si poder usar el error 1e-8 o usar el epsilon normal
        return loss
    
    def train(self, X, Y=None, epochs=1000):
        """
        Train the autoencoder
        Args:
            X: Input data
            Y: Target data
            epochs: Number of training epochs
        """
        if Y is None:
            Y = X  # Autoencoder target is the input itself

        losses = []

        log.info("Starting training for {} epochs...".format(epochs))
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")

        for epoch in pbar:

            activations, z_values = self.forward(X)
            loss = self.compute_loss(activations[-1], Y)
            losses.append(loss)
            dW, db = self.backward(X, Y, activations, z_values)
            self.update_parameters(dW, db)

            pbar.set_postfix(loss=loss)

            if epoch % 100 == 0 or epoch == epochs - 1:
                log.info("Epoch {}: Loss = {:.6f}".format(epoch, loss))

            if abs(loss) < self.epsilon:
                log.info("Convergence reached at epoch {}.".format(epoch))
                break

            # self.error_entropy_ant = loss

        return losses

    def predict(self, X):
        """Predict output for given input X."""
        A = X
        A = X
        for i in range(len(self.weights)):
            z = np.dot(A, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                A = self.activation(z)
            else:
                A = sigmoid(z)
        return A
    
    def get_latent_representation(self, X):
        """Find the representation of X in the latent space."""
        A = X
        latent_layer_idx = len(self.weights) // 2
        
        for i in range(latent_layer_idx):
            z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.activation(z)
        
        return A
    
    def decode_from_latent(self, latent):
        """Decodes de latent representation back to "input" space."""
        A = latent
        latent_layer_idx = len(self.weights) // 2
        
        for i in range(latent_layer_idx, len(self.weights)):
            z = np.dot(A, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                A = self.activation(z)
            else:
                A = sigmoid(z)
        
        return A
    
    def evaluate_pixel_error(self, X):
        """Evaluate pixel-wise error over dataset X."""
        output = self.predict(X)
        output_binary = (output > 0.5).astype(float)
        errors = np.sum(np.abs(X - output_binary), axis=1)
        return errors
    

    ## Funcion Generativa
    def generate_interpolation(self, X, idx1, idx2, alpha=0.5):
        """
        New character generation by interpolating between two characters in latent space.
        
        Args:
            X: Dataset
            idx1, idx2: Indices to interpolate
            alpha: Interpolation factor [0, 1]
        
        Returns:
            New character generated
        """
        latent = self.get_latent_representation(X)
        new_latent = alpha * latent[idx1] + (1 - alpha) * latent[idx2]
        new_latent = new_latent.reshape(1, -1)
        new_char = self.decode_from_latent(new_latent)
        return (new_char > 0.5).astype(float)
    

# =========== DEPRECADO ==========

    #⛔️solo lo guardo por si hace falta cargar desde txt⛔️


    # def save_state(self, filename):
    #     """Save model state to a file."""
        
    #     network_data = [ 
    #         {
    #             'weights': entry[0].tolist(),  # Convertir a lista de Python
    #             'bias': entry[1].tolist()       # Convertir a lista de Python
    #         }
    #         for entry in zip(self.weights, self.biases)
    #     ]
        
    #     with open("outputs/{}".format(filename), 'w') as f:
    #         import json
    #         json.dump(network_data, f)
        
    #     log.info("Model state saved to {}".format(filename))

    # # def load_network_state(self, filename):
    # #     """Carga los pesos y bias de una red neuronal desde un archivo."""
    # #     import json
        
    # #     with open(filename, 'r') as f:
    # #         network_data = json.load(f)
        
    # #     weights_list = []
    # #     biases_list = []
        
    # #     for layer in network_data:
    # #         weights_list.append(np.array(layer['weights']))
    # #         biases_list.append(np.array(layer['bias']))
        
    # #     return weights_list, biases_list
    
    # def load_network_state(self, filename):
    #     """
    #     Carga los pesos y bias de una red neuronal desde un archivo.
        
    #     Args:
    #         filename (str): Ruta al archivo con los pesos y bias
            
    #     Returns:
    #         tuple: (weights_list, biases_list) donde cada elemento es un numpy array
    #     """
    #     weights_list = []
    #     biases_list = []
        
    #     with open(filename, 'r') as f:
    #         content = f.read()
        
    #     # Dividir por líneas que contienen diccionarios completos
    #     # Usamos un patrón que identifica el inicio de cada diccionario
    #     layer_strings = re.split(r'\n(?=\{\'weights\':)', content)
    #     layer_strings = [s.strip() for s in layer_strings if s.strip()]
        
    #     for layer_str in layer_strings:
    #         try:
    #             # Evaluar el string como diccionario de Python
    #             layer_dict = eval(layer_str, {"__builtins__": {}, "array": np.array})
                
    #             # Extraer weights y bias
    #             weights = layer_dict['weights']
    #             bias = layer_dict['bias']
                
    #             # Convertir a numpy arrays si no lo son ya
    #             if not isinstance(weights, np.ndarray):
    #                 weights = np.array(weights)
    #             if not isinstance(bias, np.ndarray):
    #                 bias = np.array(bias)
                
    #             weights_list.append(weights)
    #             biases_list.append(bias)
                
    #         except Exception as e:
    #             print(f"Error procesando capa: {e}")
    #             continue
        
    #     return weights_list, biases_list
    
    # def initialize_from_file(self, filename):
    #     """Initialize model weights and biases from a file."""
    #     weights, biases = self.load_network_state(filename)
    #     self.weights = weights
    #     self.biases = biases
    #     self.n_layers = len(self.weights) + 1
    #     log.info("Model state loaded from {}".format(filename))

    def save_state_pickle(self, filename):
        """Save model state to a file."""
        import pickle
        
        network_data = {
            'weights': self.weights,
            'biases': self.biases
        }
        
        with open("outputs/{}".format(filename), 'wb') as f:
            pickle.dump(network_data, f)
        
        log.info("Model state saved to {}".format(filename))

    def initialize_from_file_pickle(self, filename):
        """Initialize model weights and biases from a file."""
        import pickle
        
        with open(filename, 'rb') as f:
            network_data = pickle.load(f)
        
        self.weights = network_data['weights']
        self.biases = network_data['biases']
        self.n_layers = len(self.weights) + 1
        log.info("Model state loaded from {}".format(filename))