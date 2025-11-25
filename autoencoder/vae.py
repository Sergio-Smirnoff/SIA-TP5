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
        n_out = self.latent_dim        # Dimensión latente (2 --> 4 neuronas)
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

        print('MU-SIGMA-EPSILON-Z')
        print(f'{self.mu} - {self.std} - {eps} - {self.z}')

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
                 activations_dec, z_values_dec, beta=1):

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

        #tiene en cuenta el beta annealing
        dmu = dmu_recon  + (dmu_kl * beta)  # (batch_size, latent_dim)
        dlogvar = dlogvar_recon  + (dlogvar_kl * beta)  # (batch_size, latent_dim)

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

    def update_parameters_sgd(self, dW_enc, db_enc, dW_mu, db_mu,
                               dW_logvar, db_logvar, dW_dec, db_dec):
        
        print(f'Inside update sgd')
        for i in range(len(self.encoder_weights)):
            self.encoder_weights[i] -= dW_enc[i] * self.learning_rate
            self.encoder_biases[i] -= db_enc[i] * self.learning_rate
        

        self.W_mu -= dW_mu * self.learning_rate
        self.b_mu -= db_mu * self.learning_rate
        
        self.W_logvar -= dW_logvar * self.learning_rate
        self.b_logvar -= db_logvar * self.learning_rate

        for i in range(len(self.decoder_weights)):
            self.decoder_weights[i] -= dW_dec[i] * self.learning_rate
            self.decoder_biases[i] -= db_dec[i] * self.learning_rate
        # print(f'Values after: \nW_mu: {self.W_mu} \nb_mu: {self.b_mu} \nW_logvar: {self.W_logvar} \nb_logvar: {self.b_logvar }')


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

    def update(self, dW_enc, db_enc, dW_mu, db_mu,
                               dW_logvar, db_logvar, dW_dec, db_dec):
        if self.optimizer =='sgd':
            self.update_parameters_sgd(dW_enc, db_enc, dW_mu, db_mu,
                               dW_logvar, db_logvar, dW_dec, db_dec)
        else:
            self.update_parameters_adam(dW_enc, db_enc, dW_mu, db_mu,
                               dW_logvar, db_logvar, dW_dec, db_dec)

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

            print(f'recon_loss={loss_recon} + KL={kl_loss} = {loss}')

            rango.set_postfix(loss=loss)

            # Backward
            # beta = epoch/epochs #---> lineal
            # beta = max(0, (3*epoch/epochs)-2)
            
            # beta = np.tanh(epoch/epochs)
            # beta = np.sin(6 *epoch/epochs)**2
            # beta = np.cos(2* epoch/epochs)**2
            beta = 0.01 / (1 - (epoch/epochs))
            grads = self.backward(X, X, acts_enc, zv_enc, acts_dec, zv_dec)
            
            self.update(*grads)
            # self.save_gradients_and_weights_to_csv(*grads, epoch)
            # print(f'dW_mu={grads[2]}\n db_mu={grads[3]}\n dW_logvar={grads[4]}\n db_logvar={grads[5]}')
            if epoch % (epochs/10) == 0 or epoch == epochs-1:
                # print(f"Epoch {epoch}: Loss={loss:.4f} Recon={loss_recon:.4f} KL={kl_loss:.4f}")
                self.plot_latent_pweaseeeeee(X, epoch)
                

    def plot_latent_pweaseeeeee(self, X, epoch, n_samples=50, labels=None):
        """
        Plotea el espacio latente mostrando la distribución estocástica
        
        Args:
            X: Dataset de entrada
            epoch: Número de época actual
            n_samples: Número de muestras a tomar de cada input para visualizar la distribución
            labels: Etiquetas para cada input (opcional)
        """
        if labels is None:
            labels = range(len(X))
        
        all_samples = []
        sample_labels = []
        mu_points = []
        
        for idx, x_input in enumerate(X):
            # Obtener mu, logvar y muestras
            mu, logvar, z_samples = self.encode(x_input, n_samples=n_samples)
            
            # Guardar mu para plotear el centro
            mu_points.append(mu[0])
            
            # Guardar todas las muestras con su etiqueta
            for z in z_samples:
                all_samples.append(z[0])
                sample_labels.append(labels[idx])
        
        all_samples = np.array(all_samples)  # Forma (n_inputs * n_samples, latent_dim)
        mu_points = np.array(mu_points)  # Forma (n_inputs, latent_dim)
        
        # Tomar solo las primeras 2 dimensiones si latent_dim > 2
        if all_samples.shape[1] >= 2:
            samples_2d = all_samples[:, :2]
            mu_2d = mu_points[:, :2]
            
            plot_latent_space_with_distributions(
                samples_2d, 
                mu_2d,
                sample_labels,
                labels,
                n_samples=n_samples,
                output_path=f"bmp_images/bit_output/latent_space_{epoch}.png"
            )
        else:
            print(f"Warning: Latent dimension is {all_samples.shape[1]}, need at least 2 for 2D plot")
        
    def encode(self, X, n_samples=1):
        """
        Codifica X y retorna mu, logvar y opcionalmente muestras de z
        
        Args:
            X: Input data
            n_samples: Número de muestras a generar del espacio latente
        
        Returns:
            mu: Media de la distribución latente
            logvar: Log-varianza de la distribución latente
            z_samples: Lista de n_samples muestras del espacio latente (si n_samples > 0)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        A = X
        for i in range(len(self.encoder_weights)):
            Z = np.dot(A, self.encoder_weights[i]) + self.encoder_biases[i]
            A = self.activation(Z)
        
        mu = np.dot(A, self.W_mu) + self.b_mu
        logvar = np.dot(A, self.W_logvar) + self.b_logvar
        
        if n_samples > 0:
            std = np.exp(0.5 * logvar)
            z_samples = []
            for _ in range(n_samples):
                eps = np.random.randn(*mu.shape)
                z = mu + std * eps
                z_samples.append(z)
            return mu, logvar, z_samples
        else:
            return mu, logvar, None

    def decode(self, z):
        """Decodifica desde el espacio latente z"""
        # if z.ndim == 1:
        #     z = z.reshape(1, -1)
        
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

    def save_gradients_and_weights_to_csv(self, dW_enc, db_enc, dW_mu, db_mu,
                                       dW_logvar, db_logvar, dW_dec, db_dec,
                                       epoch, filename_prefix='training_data'):
        """
        Guarda los gradientes y pesos de cada capa en archivos CSV.
        """
        import pandas as pd
        from pathlib import Path
        
        # Crear directorio si no existe
        Path("outputs/gradients").mkdir(parents=True, exist_ok=True)
        
        # ==================== ENCODER ====================
        for i in range(len(self.encoder_weights)):
            data = {
                'epoch': [epoch] * (self.encoder_weights[i].size + self.encoder_biases[i].size),
                'layer': [f'encoder_{i}'] * (self.encoder_weights[i].size + self.encoder_biases[i].size),
                'parameter_type': ['weight'] * self.encoder_weights[i].size + ['bias'] * self.encoder_biases[i].size,
                'gradient': np.concatenate([dW_enc[i].flatten(), db_enc[i].flatten()]),
                'value': np.concatenate([self.encoder_weights[i].flatten(), self.encoder_biases[i].flatten()])
            }
            
            df = pd.DataFrame(data)
            filepath = f"outputs/gradients/{filename_prefix}_encoder_layer_{i}.csv"
            
            # Append si el archivo existe, sino crear nuevo
            if Path(filepath).exists():
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
        
        # ==================== MU ====================
        data_mu = {
            'epoch': [epoch] * (self.W_mu.size + self.b_mu.size),
            'layer': ['mu'] * (self.W_mu.size + self.b_mu.size),
            'parameter_type': ['weight'] * self.W_mu.size + ['bias'] * self.b_mu.size,
            'gradient': np.concatenate([dW_mu.flatten(), db_mu.flatten()]),
            'value': np.concatenate([self.W_mu.flatten(), self.b_mu.flatten()])
        }
        
        df_mu = pd.DataFrame(data_mu)
        filepath_mu = f"outputs/gradients/{filename_prefix}_mu.csv"
        
        if Path(filepath_mu).exists():
            df_mu.to_csv(filepath_mu, mode='a', header=False, index=False)
        else:
            df_mu.to_csv(filepath_mu, index=False)
        
        # ==================== LOGVAR ====================
        data_logvar = {
            'epoch': [epoch] * (self.W_logvar.size + self.b_logvar.size),
            'layer': ['logvar'] * (self.W_logvar.size + self.b_logvar.size),
            'parameter_type': ['weight'] * self.W_logvar.size + ['bias'] * self.b_logvar.size,
            'gradient': np.concatenate([dW_logvar.flatten(), db_logvar.flatten()]),
            'value': np.concatenate([self.W_logvar.flatten(), self.b_logvar.flatten()])
        }
        
        df_logvar = pd.DataFrame(data_logvar)
        filepath_logvar = f"outputs/gradients/{filename_prefix}_logvar.csv"
        
        if Path(filepath_logvar).exists():
            df_logvar.to_csv(filepath_logvar, mode='a', header=False, index=False)
        else:
            df_logvar.to_csv(filepath_logvar, index=False)
        
        # ==================== DECODER ====================
        for i in range(len(self.decoder_weights)):
            data = {
                'epoch': [epoch] * (self.decoder_weights[i].size + self.decoder_biases[i].size),
                'layer': [f'decoder_{i}'] * (self.decoder_weights[i].size + self.decoder_biases[i].size),
                'parameter_type': ['weight'] * self.decoder_weights[i].size + ['bias'] * self.decoder_biases[i].size,
                'gradient': np.concatenate([dW_dec[i].flatten(), db_dec[i].flatten()]),
                'value': np.concatenate([self.decoder_weights[i].flatten(), self.decoder_biases[i].flatten()])
            }
            
            df = pd.DataFrame(data)
            filepath = f"outputs/gradients/{filename_prefix}_decoder_layer_{i}.csv"
            
            if Path(filepath).exists():
                df.to_csv(filepath, mode='a', header=False, index=False)
            else:
                df.to_csv(filepath, index=False)
        
        print(f"Gradientes y pesos guardados para época {epoch}")



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

def plot_latent_space_with_distributions(samples, mu_points, sample_labels, unique_labels, 
                                         n_samples=50, output_path='latent_space.png'):
    """
    Plot latent space mostrando las distribuciones estocásticas de cada input
    
    Args:
        samples: Array de todas las muestras (n_inputs * n_samples, 2)
        mu_points: Array de los puntos mu (n_inputs, 2)
        sample_labels: Lista con las etiquetas de cada muestra
        unique_labels: Lista de etiquetas únicas
        n_samples: Número de muestras por input
        output_path: Ruta donde guardar el plot
    """
    
    plt.figure(figsize=(14, 12))
    
    # Obtener colores únicos
    unique_labels_array = np.array(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels_array)))
    
    # Plotear las muestras (distribuciones)
    for idx, label in enumerate(unique_labels_array):
        # Filtrar muestras de este input
        mask = np.array(sample_labels) == label
        
        if np.sum(mask) > 0:
            # Plotear la nube de puntos (distribución)
            plt.scatter(samples[mask, 0], samples[mask, 1],
                       alpha=0.15,  # Transparencia alta para ver densidad
                       s=30,
                       c=[colors[idx]],
                       edgecolors='none',
                       label=None)  # No agregar a leyenda
    
    # Plotear los puntos mu (centros) encima
    for idx, label in enumerate(unique_labels_array):
        plt.scatter(mu_points[idx, 0], mu_points[idx, 1],
                   label=f'{label}' if label != '\x7f' else 'DEL',
                   alpha=1.0,
                   s=200,
                   c=[colors[idx]],
                   edgecolors='black',
                   linewidth=2,
                   marker='*',  # Estrella para destacar
                   zorder=100)  # Plotear encima de todo
        
        # Anotar con la etiqueta
        display_text = 'DEL' if label == '\x7f' else str(label)
        plt.annotate(display_text,
                    (mu_points[idx, 0], mu_points[idx, 1]),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha='center',
                    fontsize=11,
                    weight='bold',
                    bbox=dict(boxstyle='round,pad=0.4',
                             facecolor='white',
                             edgecolor=colors[idx],
                             linewidth=2,
                             alpha=0.9),
                    zorder=101)
    
    # Configurar ejes y estilo
    plt.xlabel('Latent Dimension 1 (z₁)', fontsize=12)
    plt.ylabel('Latent Dimension 2 (z₂)', fontsize=12)
    plt.title(f'Latent Space Distribution (n={n_samples} samples per input)', 
             fontsize=14, weight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Ajustar límites con margen
    x_range = samples[:, 0].max() - samples[:, 0].min()
    y_range = samples[:, 1].max() - samples[:, 1].min()
    x_margin = max(x_range * 0.15, 0.5)
    y_margin = max(y_range * 0.15, 0.5)
    
    plt.xlim(samples[:, 0].min() - x_margin, samples[:, 0].max() + x_margin)
    plt.ylim(samples[:, 1].min() - y_margin, samples[:, 1].max() + y_margin)
    
    # Agregar líneas en 0 para referencia
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Leyenda
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
              ncol=1, fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    
    print(f"Saving distribution plot to: {output_path}")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print("Distribution plot saved successfully")

   