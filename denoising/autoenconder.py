
import sys
import logging as log
import numpy as np
from tqdm import tqdm
import re
from activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative

def add_salt_and_pepper_noise(X, noise_amount=0.10):
        """
        X: matriz binaria (0 o 1) de shape (n_samples, n_features)
        noise_amount: porcentaje de píxeles a alterar
        
        Retorna: X_noisy
        """
        X_noisy = X.copy()

        # número de píxeles a alterar por muestra
        n_features = X.shape[1]
        num_noisy_pixels = int(noise_amount * n_features)

        for i in range(X.shape[0]):
            idx = np.random.choice(n_features, num_noisy_pixels, replace=False)

            # Salt & Pepper: valores al azar 0 o 1
            X_noisy[i, idx] = 1 - X_noisy[i, idx]

        return X_noisy

def add_gaussian_noise(X, sigma=0.0):
    """
    Añade ruido gaussiano a datos binarios en rango 0–1.
    
    X: matriz (n_samples, n_features)
    sigma: desviación estándar del ruido
    """
    noise = np.random.normal(0, sigma, X.shape)
    X_noisy = X + noise
    X_noisy = np.clip(X_noisy, 0.0, 1.0)
    return X_noisy

class BasicAutoencoder:

    def __init__(
            self,
            # ej: [input_size, l1_hidden_size, l2_hidden_size] 
            # l1 == l2, input_size == output_size, l2 == 2 as middle layer always 2
            architecture=[35, 16, 8, 4, 2], 
            learning_rate=0.01,
            epsilon=1e-4,
            optimizer='sgd',
            activation_function='sigmoid',
            noise_amount=0.0,
            sigma=0.0,
            seed=42
            ):

        log.info("Initializing Autoencoder...")
        # Hyperparameters
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.optimizer = optimizer

       # Additional parameters
        self.seed = seed
        self.noise_amount=noise_amount
        self.sigma=sigma

        # Errors
        self.error_entropy = 1
        self.error_entropy_ant = np.inf
        self.error_entropy_min = np.inf

        self.error_mse = 1
        self.error_mse_ant = np.inf
        self.error_mse_min = np.inf

        if self.optimizer == 'adam':
            log.info("Adam optimizer selected")
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon_adam = 1e-8
            self.t = 0

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

        if self.optimizer == 'adam':
            # Initialize Adam parameters
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]

        log.info("Weights and biases initialized.")

    def normalize(self, X):
        # Normalización a [0,1]
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        return (X - X_min) / (X_max - X_min + 1e-8), X_min, X_max

        
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

        # BCE + sigmoid simplificación: dZ = A_out - Y
        dZ = activations[-1] - Y

        for i in reversed(range(len(self.weights))):

            dW[i] = np.dot(activations[i].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self.activation_derivative(z_values[i - 1])

        return dW, db

    def update_parameters(self, dW, db):
        """
        Update weights and biases using gradients
        Using SGD optimizer
        Args:
            dW: Gradients for weights
            db: Gradients for biases
        """
        log.info("Updating parameters...")
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
        log.info("Parameters updated.")

    def update_parameters_adam(self, dW, db):

        self.t += 1  # contador de steps para corrección de sesgo

        for i in range(self.n_layers - 1):

            # --- MOMENTOS DE W ---
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * dW[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (dW[i] ** 2)

            # Corrección por sesgo
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)

            # Update final
            self.weights[i] -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + 1e-8)

            # --- MOMENTOS DE b ---
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db[i] ** 2)
            # Corrección por sesgo
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Update final
            self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + 1e-8)

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
        ## threadhold

        Y_pred_binary = (Y_pred >= 0.5).astype(float)

        #loss = -np.mean(Y_true * np.log(Y_pred + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred + 1e-8)) 
        loss = -np.mean(Y_true * np.log(Y_pred_binary + 1e-8) + (1 - Y_true) * np.log(1 - Y_pred_binary + 1e-8))
        # TODO: verificar si poder usar el error 1e-8 o usar el epsilon normal
        return loss
    
    def train(self, X, Y=None, epochs=1000):

        X_norm, X_min, X_max = self.normalize(X)
        X_norm = X

        if Y is None:
            Y = X_norm

        losses = []
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        
        for epoch in pbar:
            if self.noise_amount != 0.0:
                X_noisy = add_salt_and_pepper_noise(X_norm, self.noise_amount)
            if self.sigma != 0.0:
                X_noisy = add_gaussian_noise(X_norm, self.sigma)

            activations, z_values = self.forward(X_noisy)
            loss = self.compute_loss(activations[-1], Y)
            dW, db = self.backward(X_noisy, Y, activations, z_values)
            
            if self.optimizer == 'adam':
                self.update_parameters_adam(dW, db)
            else:
                self.update_parameters(dW, db)
            pbar.set_postfix(loss=loss)
            losses.append(loss)
            log.info("Epoch {}/{} - Loss: {:.6f}".format(epoch + 1, epochs, loss))
            if abs(loss) < self.epsilon: 
                log.info("Convergence reached at epoch {}.".format(epoch)) 
                break 
            self.error_entropy_ant = loss
        self.X_min = X_min
        self.X_max = X_max

        return losses


    def predict(self, X):
        #X_norm = (X - self.X_min) / (self.X_max - self.X_min + 1e-8)
        activations, _ = self.forward(X)
        Y_norm = activations[-1]
        Y_pred = Y_norm * (self.X_max - self.X_min) + self.X_min
        return (Y_pred >= 0.5).astype(float)

    
    def get_latent_representation(self, X):
        """Find the representation of X in the latent space."""
        A = X
        for i in range(len(self.weights) // 2):
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