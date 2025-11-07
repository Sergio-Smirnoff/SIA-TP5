
import sys
import logging as log
import numpy as np
from autoencoder.activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative


class BasicAutoencoder:

    def __init__(
            self,
            # ej: [input_size, l1_hidden_size, l2_hidden_size, l3_hidden_size, output_size] 
            # l1 == l2, input_size == output_size, l2 == 2 as middle layer always 2
            architecture, 
            learning_rate=0.01,
            epsilon=1e-4,
            optimizer='sgd',
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

        # Errores
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

        log.info("Autoencoder initialized with architecture: {}".format(architecture))


    def wb_initializer(self, architecture, seed):

        self.seed = seed
        np.random.seed(seed) if seed is not None else None

        self.n_layers = len(architecture)
        self.weights = []
        self.biases = []
        log.info("Initializing weights and biases...")
        for i in range(self.n_layers - 1):
            # Xavier: W ~ U(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))
            limit = np.sqrt(6 / (architecture[i] + architecture[i+1]))
            W = np.random.uniform(-limit, limit, 
                                 (architecture[i], architecture[i+1]))
            b = np.zeros((1, architecture[i+1]))
            
            self.weights.append(W)
            self.biases.append(b)

        log.info("Weights and biases initialized.")
        
    def forward(self, X):
        """
        Args:
            X: input data
        Returns:
            activations: List of activations per layer
            z_values: List of Z values per layer
        """
        log.info("Performing forward propagation...")
        activations = [X]
        z_values = []
        
        A = X
        for i in range(self.n_layers - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.activation(Z)

            z_values.append(Z)
            activations.append(A)
            log.debug("Layer {}: Z: {}, A: {}".format(i+1, Z, A))
        
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
        m = X.shape[0]
        
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Error en la salida
        dA = activations[-1] - Y
        
        # Backpropagation
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dz = dA * sigmoid_derivative(z_values[i])
            else:
                dz = dA * self.activation_derivative(z_values[i])

            dW[i] = np.dot(activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dz, self.weights[i].T)
        
        return dW, db
