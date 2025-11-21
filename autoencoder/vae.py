from autoenconder import BasicAutoencoder
import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative
from tqdm import tqdm
import matplotlib.pyplot as plt

class VAE(BasicAutoencoder):
    """Variational Autoencoder implementation."""

    def __init__(self, architecture=[35, 16, 8, 4], learning_rate=0.01, epsilon=0.0001, optimizer='sgd', activation_function='sigmoid', seed=42):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.seed = seed
        
        if optimizer == "adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.eps_adam = 1e-8
            self.t = 0
        
        # Activación
        if activation_function == "relu":
            self.activation = relu
            self.activation_derivative = relu_derivative
        elif activation_function == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        else:
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        
        # ----------------------------------------------------
        #               Inicialización VAE
        # ----------------------------------------------------
        np.random.seed(seed)

        # Architecture: encoder + latent + decoder
        self.encoder_arch = architecture[:-1]  # [35, 16, 8]
        self.latent_dim = architecture[-1]      # 2
        self.decoder_arch = architecture[::-1]  # [2, 8, 16, 35]

        print(f'Encoder arch: {self.encoder_arch}') 
        print(f'Decoder arch: {self.decoder_arch}')
        print(f'Latent dim: {self.latent_dim}')

        # ---------- inicializar encoder ----------
        self.encoder_weights = []
        self.encoder_biases = []
        for i in range(len(self.encoder_arch) - 1):
            n_in = self.encoder_arch[i]
            n_out = self.encoder_arch[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((1, n_out))
            self.encoder_weights.append(W)
            self.encoder_biases.append(b)

        # ---------- inicializar μ y logvar ----------
        # Una sola matriz que proyecta de encoder_arch[-1] a latent_dim
        n_in = self.encoder_arch[-1]  # Última capa del encoder (8)
        n_out = self.latent_dim        # Dimensión latente (2)
        limit = np.sqrt(6 / (n_in + n_out))
        
        self.W_mu = np.random.uniform(-limit, limit, (n_in, n_out))
        self.b_mu = np.zeros((1, n_out))
        
        self.W_logvar = np.random.uniform(-limit, limit, (n_in, n_out))
        self.b_logvar = np.zeros((1, n_out))

        # ---------- inicializar decoder ----------
        self.decoder_weights = []
        self.decoder_biases = []
        for i in range(len(self.decoder_arch) - 1):
            n_in = self.decoder_arch[i]
            n_out = self.decoder_arch[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))
            W = np.random.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((1, n_out))
            self.decoder_weights.append(W)
            self.decoder_biases.append(b)

        # ----------------------------------------------------
        #              Adam moments
        # ----------------------------------------------------
        if self.optimizer == "adam":
            # Encoder
            self.m_w_enc = [np.zeros_like(w) for w in self.encoder_weights]
            self.v_w_enc = [np.zeros_like(w) for w in self.encoder_weights]
            self.m_b_enc = [np.zeros_like(b) for b in self.encoder_biases]
            self.v_b_enc = [np.zeros_like(b) for b in self.encoder_biases]

            # Mu
            self.m_w_mu = np.zeros_like(self.W_mu)
            self.v_w_mu = np.zeros_like(self.W_mu)
            self.m_b_mu = np.zeros_like(self.b_mu)
            self.v_b_mu = np.zeros_like(self.b_mu)

            # Logvar
            self.m_w_logvar = np.zeros_like(self.W_logvar)
            self.v_w_logvar = np.zeros_like(self.W_logvar)
            self.m_b_logvar = np.zeros_like(self.b_logvar)
            self.v_b_logvar = np.zeros_like(self.b_logvar)

            # Decoder
            self.m_w_dec = [np.zeros_like(w) for w in self.decoder_weights]
            self.v_w_dec = [np.zeros_like(w) for w in self.decoder_weights]
            self.m_b_dec = [np.zeros_like(b) for b in self.decoder_biases]
            self.v_b_dec = [np.zeros_like(b) for b in self.decoder_biases]


    def forward(self, X):
        activations_enc = [X]
        z_values_enc = []

        A = X
        # ------- Encoder forward -------
        for i in range(len(self.encoder_weights)):
            Z = np.dot(A, self.encoder_weights[i]) + self.encoder_biases[i]
            A = self.activation(Z)
            z_values_enc.append(Z)
            activations_enc.append(A)

        # ------- Latent layer -------
        # A tiene forma (batch_size, encoder_arch[-1])
        # mu y logvar tendrán forma (batch_size, latent_dim)
        self.mu = np.dot(A, self.W_mu) + self.b_mu
        self.logvar = np.dot(A, self.W_logvar) + self.b_logvar
        self.std = np.exp(0.5 * self.logvar)

        # eps tiene la misma forma que mu: (batch_size, latent_dim)
        eps = np.random.randn(*self.mu.shape)
        self.z = self.mu + self.std * eps   # reparam trick

        # ------- Decoder forward -------
        activations_dec = [self.z]
        z_values_dec = []
        A = self.z
        for i in range(len(self.decoder_weights)):
            Z = np.dot(A, self.decoder_weights[i]) + self.decoder_biases[i]
            if i < len(self.decoder_weights) - 1:
                A = self.activation(Z)
            else:
                A = sigmoid(Z)
            z_values_dec.append(Z)
            activations_dec.append(A)

        return activations_enc, z_values_enc, activations_dec, z_values_dec
        
    def backward(self, X, Y,
                 activations_enc, z_values_enc,
                 activations_dec, z_values_dec):

        m = X.shape[0]  # batch size

        # --------------------------
        # 1. DECODER BACKPROP
        # --------------------------
        dW_dec = [np.zeros_like(W) for W in self.decoder_weights]
        db_dec = [np.zeros_like(b) for b in self.decoder_biases]

        # Gradiente de la loss de reconstrucción respecto a la salida
        dZ = (activations_dec[-1] - Y)  # BCE + sigmoid simplificado

        for i in reversed(range(len(self.decoder_weights))):
            A_prev = activations_dec[i]
            dW_dec[i] = np.dot(A_prev.T, dZ) / m
            db_dec[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.decoder_weights[i].T)
                dZ = dA_prev * self.activation_derivative(z_values_dec[i - 1])

        # Gradiente que llega a z desde el decoder
        # dZ tiene forma (batch_size, decoder_arch[0]) = (batch_size, latent_dim)
        dZ_latent = np.dot(dZ, self.decoder_weights[0].T)

        # --------------------------
        # 2. GRADIENTES DE MU Y LOGVAR
        # --------------------------

        # Gradiente de la loss de reconstrucción por reparameterización
        # z = mu + std * eps  =>  eps = (z - mu) / std
        eps = (self.z - self.mu) / (self.std + 1e-8)

        dmu_recon = dZ_latent  # (batch_size, latent_dim)
        dstd = dZ_latent * eps  # (batch_size, latent_dim)
        # d(std)/d(logvar) = d(exp(0.5*logvar))/d(logvar) = 0.5 * exp(0.5*logvar) = 0.5 * std
        dlogvar_recon = dstd * (0.5 * self.std)

        # Gradientes de la KL divergence
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # dKL/dmu = mu
        # dKL/dlogvar = 0.5 * (exp(logvar) - 1)
        dmu_kl = self.mu  # (batch_size, latent_dim)
        dlogvar_kl = 0.5 * (np.exp(self.logvar) - 1)  # (batch_size, latent_dim)

        # Gradientes totales
        dmu = dmu_recon + dmu_kl  # (batch_size, latent_dim)
        dlogvar = dlogvar_recon + dlogvar_kl  # (batch_size, latent_dim)

        # Gradientes de los pesos y biases de mu y logvar
        A_enc_last = activations_enc[-1]  # (batch_size, encoder_arch[-1])

        # dW_mu: (encoder_arch[-1], latent_dim)
        dW_mu = np.dot(A_enc_last.T, dmu) / m
        db_mu = np.sum(dmu, axis=0, keepdims=True) / m

        # dW_logvar: (encoder_arch[-1], latent_dim)
        dW_logvar = np.dot(A_enc_last.T, dlogvar) / m
        db_logvar = np.sum(dlogvar, axis=0, keepdims=True) / m

        # --------------------------
        # 3. PROPAGAR HACIA EL ENCODER
        # --------------------------
        # Gradiente que llega a la última activación del encoder
        dh = (
            np.dot(dmu, self.W_mu.T) +
            np.dot(dlogvar, self.W_logvar.T)
        )  # (batch_size, encoder_arch[-1])

        dW_enc = [np.zeros_like(W) for W in self.encoder_weights]
        db_enc = [np.zeros_like(b) for b in self.encoder_biases]

        dZ = dh
        for i in reversed(range(len(self.encoder_weights))):
            A_prev = activations_enc[i]
            dW_enc[i] = np.dot(A_prev.T, dZ) / m
            db_enc[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.encoder_weights[i].T)
                dZ = dA_prev * self.activation_derivative(z_values_enc[i - 1])

        return dW_enc, db_enc, dW_mu, db_mu, dW_logvar, db_logvar, dW_dec, db_dec

    def update_parameters_adam(self,
                               dW_enc, db_enc,
                               dW_mu, db_mu,
                               dW_logvar, db_logvar,
                               dW_dec, db_dec):

        self.t += 1
        eps = 1e-8

        # ===========================================================
        #                    ENCODER
        # ===========================================================
        for i in range(len(self.encoder_weights)):
            # Pesos
            self.m_w_enc[i] = self.beta1 * self.m_w_enc[i] + (1 - self.beta1) * dW_enc[i]
            self.v_w_enc[i] = self.beta2 * self.v_w_enc[i] + (1 - self.beta2) * (dW_enc[i] ** 2)

            m_hat = self.m_w_enc[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w_enc[i] / (1 - self.beta2 ** self.t)

            self.encoder_weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # Bias
            self.m_b_enc[i] = self.beta1 * self.m_b_enc[i] + (1 - self.beta1) * db_enc[i]
            self.v_b_enc[i] = self.beta2 * self.v_b_enc[i] + (1 - self.beta2) * (db_enc[i] ** 2)

            m_hat = self.m_b_enc[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b_enc[i] / (1 - self.beta2 ** self.t)

            self.encoder_biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        # ===========================================================
        #                    MU
        # ===========================================================
        self.m_w_mu = self.beta1 * self.m_w_mu + (1 - self.beta1) * dW_mu
        self.v_w_mu = self.beta2 * self.v_w_mu + (1 - self.beta2) * (dW_mu ** 2)

        m_hat = self.m_w_mu / (1 - self.beta1 ** self.t)
        v_hat = self.v_w_mu / (1 - self.beta2 ** self.t)

        self.W_mu -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.m_b_mu = self.beta1 * self.m_b_mu + (1 - self.beta1) * db_mu
        self.v_b_mu = self.beta2 * self.v_b_mu + (1 - self.beta2) * (db_mu ** 2)

        m_hat = self.m_b_mu / (1 - self.beta1 ** self.t)
        v_hat = self.v_b_mu / (1 - self.beta2 ** self.t)

        self.b_mu -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        # ===========================================================
        #                    LOGVAR
        # ===========================================================
        self.m_w_logvar = self.beta1 * self.m_w_logvar + (1 - self.beta1) * dW_logvar
        self.v_w_logvar = self.beta2 * self.v_w_logvar + (1 - self.beta2) * (dW_logvar ** 2)

        m_hat = self.m_w_logvar / (1 - self.beta1 ** self.t)
        v_hat = self.v_w_logvar / (1 - self.beta2 ** self.t)

        self.W_logvar -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        self.m_b_logvar = self.beta1 * self.m_b_logvar + (1 - self.beta1) * db_logvar
        self.v_b_logvar = self.beta2 * self.v_b_logvar + (1 - self.beta2) * (db_logvar ** 2)

        m_hat = self.m_b_logvar / (1 - self.beta1 ** self.t)
        v_hat = self.v_b_logvar / (1 - self.beta2 ** self.t)

        self.b_logvar -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        # ===========================================================
        #                    DECODER
        # ===========================================================
        for i in range(len(self.decoder_weights)):
            # Pesos
            self.m_w_dec[i] = self.beta1 * self.m_w_dec[i] + (1 - self.beta1) * dW_dec[i]
            self.v_w_dec[i] = self.beta2 * self.v_w_dec[i] + (1 - self.beta2) * (dW_dec[i] ** 2)

            m_hat = self.m_w_dec[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w_dec[i] / (1 - self.beta2 ** self.t)

            self.decoder_weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # Bias
            self.m_b_dec[i] = self.beta1 * self.m_b_dec[i] + (1 - self.beta1) * db_dec[i]
            self.v_b_dec[i] = self.beta2 * self.v_b_dec[i] + (1 - self.beta2) * (db_dec[i] ** 2)

            m_hat = self.m_b_dec[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b_dec[i] / (1 - self.beta2 ** self.t)

            self.decoder_biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)


    def train(self, X, Y=None, epochs=1000):

        rango = tqdm(range(epochs), desc="Training")
        for epoch in rango:
            acts_enc, zv_enc, acts_dec, zv_dec = self.forward(X)
            Y_pred = acts_dec[-1]

            # Reconstruction loss (BCE)
            loss_recon = -np.mean(X * np.log(Y_pred + 1e-8) + (1-X)*np.log(1-Y_pred+1e-8))

            # KL loss
            kl_loss = -0.5 * np.mean(1 + self.logvar - self.mu**2 - np.exp(self.logvar))

            loss = loss_recon + kl_loss

            rango.set_postfix(loss=loss)

            # Backward
            grads = self.backward(X, X, acts_enc, zv_enc, acts_dec, zv_dec)
            self.update_parameters_adam(*grads)

            if epoch % (epochs/10) == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f} Recon={loss_recon:.4f} KL={kl_loss:.4f}")
                self.plot_latent_pweaseeeeee(X, epoch)
                

    def plot_latent_pweaseeeeee(self, X, epoch):
        latent_points = []
        for emoji in X:
            mu, logvar, z = self.encode(emoji)
            print()
            latent_points.append(z[0])
        latent_points = np.array(latent_points)  # Forma (10, 4)
        latent_2d = latent_points[:, :2]  # Forma (10, 2)
        plot_latent_space(latent_2d, 
            X,
            labels=range(len(X)),
            output_path=f"bmp_images/bit_output/latent_space_{epoch}.png")
        
    def encode(self, X):
        """Codifica X y retorna mu, logvar y z"""
        # Asegurar que X sea 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        activations_enc = [X]
        A = X
        
        # Forward pass por el encoder
        for i in range(len(self.encoder_weights)):
            Z = np.dot(A, self.encoder_weights[i]) + self.encoder_biases[i]
            A = self.activation(Z)
            activations_enc.append(A)
        
        # Calcular mu, logvar y z
        mu = np.dot(A, self.W_mu) + self.b_mu
        logvar = np.dot(A, self.W_logvar) + self.b_logvar
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        z = mu + std * eps
        
        return mu, logvar, z
    
    def decode(self, z):
        """Decodifica desde el espacio latente z"""
        A = z
        for i in range(len(self.decoder_weights)):
            Z = np.dot(A, self.decoder_weights[i]) + self.decoder_biases[i]
            if i < len(self.decoder_weights) - 1:
                A = self.activation(Z)
            else:
                A = sigmoid(Z)
        return A
    
    def save_state_pickle(self, filename):
        """Save VAE model state to a file."""
        import pickle
        from pathlib import Path
        
        # Crear el directorio outputs si no existe
        Path("outputs").mkdir(exist_ok=True)
        
        vae_data = {
            # Arquitectura
            'encoder_arch': self.encoder_arch,
            'latent_dim': self.latent_dim,
            'decoder_arch': self.decoder_arch,
            
            # Hiperparámetros
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'optimizer': self.optimizer,
            'seed': self.seed,
            
            # Pesos y biases del encoder
            'encoder_weights': self.encoder_weights,
            'encoder_biases': self.encoder_biases,
            
            # Pesos y biases de mu y logvar
            'W_mu': self.W_mu,
            'b_mu': self.b_mu,
            'W_logvar': self.W_logvar,
            'b_logvar': self.b_logvar,
            
            # Pesos y biases del decoder
            'decoder_weights': self.decoder_weights,
            'decoder_biases': self.decoder_biases,
        }
        
        # Si está usando Adam, guardar los momentos también
        if self.optimizer == "adam":
            vae_data.update({
                'beta1': self.beta1,
                'beta2': self.beta2,
                'eps_adam': self.eps_adam,
                't': self.t,
                # Momentos del encoder
                'm_w_enc': self.m_w_enc,
                'v_w_enc': self.v_w_enc,
                'm_b_enc': self.m_b_enc,
                'v_b_enc': self.v_b_enc,
                # Momentos de mu
                'm_w_mu': self.m_w_mu,
                'v_w_mu': self.v_w_mu,
                'm_b_mu': self.m_b_mu,
                'v_b_mu': self.v_b_mu,
                # Momentos de logvar
                'm_w_logvar': self.m_w_logvar,
                'v_w_logvar': self.v_w_logvar,
                'm_b_logvar': self.m_b_logvar,
                'v_b_logvar': self.v_b_logvar,
                # Momentos del decoder
                'm_w_dec': self.m_w_dec,
                'v_w_dec': self.v_w_dec,
                'm_b_dec': self.m_b_dec,
                'v_b_dec': self.v_b_dec,
            })
        
        filepath = f"{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(vae_data, f)
        
        print(f"VAE model state saved to {filepath}")

    def load_state_pickle(self, filename):
        """Load VAE model state from a file."""
        import pickle
        
        with open(filename, 'rb') as f:
            vae_data = pickle.load(f)
        
        # Restaurar arquitectura
        self.encoder_arch = vae_data['encoder_arch']
        self.latent_dim = vae_data['latent_dim']
        self.decoder_arch = vae_data['decoder_arch']
        
        # Restaurar hiperparámetros
        self.learning_rate = vae_data['learning_rate']
        self.epsilon = vae_data['epsilon']
        self.optimizer = vae_data['optimizer']
        self.seed = vae_data['seed']
        
        # Restaurar pesos y biases del encoder
        self.encoder_weights = vae_data['encoder_weights']
        self.encoder_biases = vae_data['encoder_biases']
        
        # Restaurar pesos y biases de mu y logvar
        self.W_mu = vae_data['W_mu']
        self.b_mu = vae_data['b_mu']
        self.W_logvar = vae_data['W_logvar']
        self.b_logvar = vae_data['b_logvar']
        
        # Restaurar pesos y biases del decoder
        self.decoder_weights = vae_data['decoder_weights']
        self.decoder_biases = vae_data['decoder_biases']
        
        # Si estaba usando Adam, restaurar los momentos
        if self.optimizer == "adam" and 't' in vae_data:
            self.beta1 = vae_data['beta1']
            self.beta2 = vae_data['beta2']
            self.eps_adam = vae_data['eps_adam']
            self.t = vae_data['t']
            
            self.m_w_enc = vae_data['m_w_enc']
            self.v_w_enc = vae_data['v_w_enc']
            self.m_b_enc = vae_data['m_b_enc']
            self.v_b_enc = vae_data['v_b_enc']
            
            self.m_w_mu = vae_data['m_w_mu']
            self.v_w_mu = vae_data['v_w_mu']
            self.m_b_mu = vae_data['m_b_mu']
            self.v_b_mu = vae_data['v_b_mu']
            
            self.m_w_logvar = vae_data['m_w_logvar']
            self.v_w_logvar = vae_data['v_w_logvar']
            self.m_b_logvar = vae_data['m_b_logvar']
            self.v_b_logvar = vae_data['v_b_logvar']
            
            self.m_w_dec = vae_data['m_w_dec']
            self.v_w_dec = vae_data['v_w_dec']
            self.m_b_dec = vae_data['m_b_dec']
            self.v_b_dec = vae_data['v_b_dec']
        
        print(f"VAE model state loaded from {filename}")



def plot_latent_space(latent, X, labels=None, output_path='latent_space.png'):
    """Plot latent space representations in 2D."""
    
    if latent.shape[1] != 2:
        raise ValueError(f"Latent space must be 2D, got {latent.shape[1]} dimensions")
    
    plt.figure(figsize=(12, 10))
    
    if labels is not None:

        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = (labels == label)
            
            if np.sum(mask) > 0:
                plt.scatter(latent[mask, 0], latent[mask, 1], 
                           label=f'{label}' if label != '\x7f' else 'DEL', 
                           alpha=0.7, 
                           s=150,
                           c=[colors[idx]],
                           edgecolors='k',
                           linewidth=0.5)
                
        for i, label in enumerate(labels):
            display_text = 'DEL' if label == '\x7f' else label
            plt.annotate(display_text, 
                        (latent[i, 0], latent[i, 1]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha='center',
                        fontsize=9,
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='white', 
                                 edgecolor='gray', 
                                 alpha=0.7))
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  ncol=2, fontsize=8)
        
    else:
        print("No labels - plotting all points")
        plt.scatter(latent[:, 0], latent[:, 1], alpha=0.7, s=100, c='blue')

    x_range = latent[:, 0].max() - latent[:, 0].min()
    y_range = latent[:, 1].max() - latent[:, 1].min()
    x_margin = max(x_range * 0.15, 0.1)
    y_margin = max(y_range * 0.15, 0.1)
    
    plt.xlim(latent[:, 0].min() - x_margin, latent[:, 0].max() + x_margin)
    plt.ylim(latent[:, 1].min() - y_margin, latent[:, 1].max() + y_margin)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Representation')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print(f"Saving to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Plot saved successfully")