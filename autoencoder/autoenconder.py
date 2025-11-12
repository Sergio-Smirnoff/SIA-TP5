
import sys
import logging as log
import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative


class BasicAutoencoder:

    def __init__(
            self,
            # ej: [input_size, l1_hidden_size, l2_hidden_size] 
            # l1 == l2, input_size == output_size, l2 == 2 as middle layer always 2
            architecture=[35, 16, 8, 2], 
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
        
        # Error at output layer
        dA = activations[-1] - Y
        
        # Backpropagation
        for i in reversed(range(len(self.weights))):
            if i == len(self.weights) - 1:
                dz = dA * sigmoid_derivative(z_values[i])
            else:
                dz = dA * self.activation_derivative(z_values[i])


            log.debug("Layer {}: dz shape: {}".format(i+1, dz.shape))
            log.debug("Layer {}: T shape: {}".format(i+1, activations[i].T.shape))

            dW[i] = np.dot(activations[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(dz, self.weights[i].T)
        log.info("Backward propagation completed.")
        return dW, db

    def update_parameters(self, dW, db):
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
        for epoch in range(epochs):
            activations, z_values = self.forward(X)
            loss = self.compute_loss(activations[-1], Y)
            losses.append(loss)
            dW, db = self.backward(X, Y, activations, z_values)
            self.update_parameters(dW, db)

            if epoch % 100 == 0 or epoch == epochs - 1:
                log.info("Epoch {}: Loss = {:.6f}".format(epoch, loss))
            if abs(self.error_entropy_ant - loss) < self.epsilon:
                log.info("Convergence reached at epoch {}.".format(epoch))
                break
        
        log.info("Training completed.")

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