from autoenconder import BasicAutoencoder
import numpy as np
from activation_functions import sigmoid, sigmoid_derivative, relu, relu_derivative, tanh, tanh_derivative
import tqdm

class VAE(BasicAutoencoder):
    """Variational Autoencoder implementation."""

    def __init__(self, architecture=[35, 16, 8, 4, 2], learning_rate=0.01, epsilon=0.0001, optimizer='sgd', activation_function='sigmoid', seed=42):
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

        # Architecture: encoder + decoder
        self.encoder_arch = architecture[:-1]
        latent_dim = architecture[-1]
        self.decoder_arch = architecture[::-1]

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

        # ---------- inicializar mu ----------
        n_in = self.encoder_arch[-1]
        limit = np.sqrt(6 / (n_in + latent_dim))
        self.W_mu = np.random.uniform(-limit, limit, (n_in, latent_dim))
        self.b_mu = np.zeros((1, latent_dim))

        # ---------- inicializar logvar ----------
        self.W_logvar = np.random.uniform(-limit, limit, (n_in, latent_dim))
        self.b_logvar = np.zeros((1, latent_dim))

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



    def wb_initializer(self, architecture, seed):
        self.architecture[-1] = 2 
        super.wb_initializer(architecture, seed)

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
        self.mu = np.dot(A, self.W_mu) + self.b_mu
        self.logvar = np.dot(A, self.W_logvar) + self.b_logvar
        self.std = np.exp(0.5 * self.logvar)

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

        m = X.shape[0]

        # --------------------------
        # 1. DECODER BACKPROP
        # --------------------------
        dW_dec = [np.zeros_like(W) for W in self.decoder_weights]
        db_dec = [np.zeros_like(b) for b in self.decoder_biases]

        dZ = (activations_dec[-1] - Y)  # BCE + sigmoid simplificado

        for i in reversed(range(len(self.decoder_weights))):
            A_prev = activations_dec[i]
            dW_dec[i] = np.dot(A_prev.T, dZ) / m
            db_dec[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.decoder_weights[i].T)
                dZ = dA_prev * self.activation_derivative(z_values_dec[i - 1])

        # ESTE GRADIENTE ES dL/dz (entra al encoder)
        dZ_latent = np.dot(dZ, self.decoder_weights[0].T)

        # --------------------------
        # 2. GRADIENTES DE MU Y LOGVAR
        # --------------------------

        # Por reparameterization:
        # z = mu + std * eps
        eps = (self.z - self.mu) / (self.std + 1e-8)

        dmu_reparam = dZ_latent
        dstd = dZ_latent * eps
        dlogvar_reparam = dstd * (0.5 * self.std)

        # KL Gradients
        dmu_kl = self.mu
        dlogvar_kl = 0.5 * (np.exp(self.logvar) - 1)

        dmu = dmu_reparam + dmu_kl
        dlogvar = dlogvar_reparam + dlogvar_kl

        # Gradientes de las capas lineales mu, logvar
        A_enc_last = activations_enc[-1]

        dW_mu = np.dot(A_enc_last.T, dmu) / m
        db_mu = np.sum(dmu, axis=0, keepdims=True) / m

        dW_logvar = np.dot(A_enc_last.T, dlogvar) / m
        db_logvar = np.sum(dlogvar, axis=0, keepdims=True) / m

        # --------------------------
        # 3. PROPAGAR HACIA EL ENCODER
        # --------------------------
        dh = (
            np.dot(dmu, self.W_mu.T) +
            np.dot(dlogvar, self.W_logvar.T)
        )

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

        self.t += 1  # incremento del contador Adam

        eps = 1e-8

        # ===========================================================
        #                    ENCODER (todas sus capas)
        # ===========================================================
        for i in range(len(self.encoder_weights)):

            # ----- Pesos -----
            self.m_w_enc[i] = self.beta1 * self.m_w_enc[i] + (1 - self.beta1) * dW_enc[i]
            self.v_w_enc[i] = self.beta2 * self.v_w_enc[i] + (1 - self.beta2) * (dW_enc[i] ** 2)

            m_hat = self.m_w_enc[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w_enc[i] / (1 - self.beta2 ** self.t)

            self.encoder_weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # ----- Bias -----
            self.m_b_enc[i] = self.beta1 * self.m_b_enc[i] + (1 - self.beta1) * db_enc[i]
            self.v_b_enc[i] = self.beta2 * self.v_b_enc[i] + (1 - self.beta2) * (db_enc[i] ** 2)

            m_hat = self.m_b_enc[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b_enc[i] / (1 - self.beta2 ** self.t)

            self.encoder_biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)


        # ===========================================================
        #                    μ  (linear_mu)
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
        #                    logvar  (linear_logvar)
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
        #                    DECODER (todas sus capas)
        # ===========================================================

        for i in range(len(self.decoder_weights)):

            # ----- Pesos -----
            self.m_w_dec[i] = self.beta1 * self.m_w_dec[i] + (1 - self.beta1) * dW_dec[i]
            self.v_w_dec[i] = self.beta2 * self.v_w_dec[i] + (1 - self.beta2) * (dW_dec[i] ** 2)

            m_hat = self.m_w_dec[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_w_dec[i] / (1 - self.beta2 ** self.t)

            self.decoder_weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            # ----- Bias -----
            self.m_b_dec[i] = self.beta1 * self.m_b_dec[i] + (1 - self.beta1) * db_dec[i]
            self.v_b_dec[i] = self.beta2 * self.v_b_dec[i] + (1 - self.beta2) * (db_dec[i] ** 2)

            m_hat = self.m_b_dec[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v_b_dec[i] / (1 - self.beta2 ** self.t)

            self.decoder_biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, X, Y=None, epochs=1000):

        # X_norm, X_min, X_max = self.normalize(X)
        # X_norm = X

        # if Y is None:
        #     Y = X_norm

        # losses = []
        # pbar = tqdm(range(epochs), desc="Training", unit="epoch")

        # for epoch in pbar:
        #     activations, z_values = self.forward(X_norm)
        #     loss = self.compute_loss(activations[-1], Y)
        #     dW, db = self.backward(X_norm, Y, activations, z_values)
            
        #     if self.optimizer == 'adam':
        #         self.update_parameters_adam(dW, db)
        #     else:
        #         self.update_parameters(dW, db)
        #     pbar.set_postfix(loss=loss)
        #     losses.append(loss)

        #     if abs(loss) < self.epsilon: 

        #         break 
        #     self.error_entropy_ant = loss
        for epoch in range(epochs):
            acts_enc, zv_enc, acts_dec, zv_dec = self.forward(X)
            Y_pred = acts_dec[-1]

            # Reconstruction loss (BCE)
            loss_recon = -np.mean(X * np.log(Y_pred + 1e-8) + (1-X)*np.log(1-Y_pred+1e-8))

            # KL loss
            kl_loss = -0.5 * np.mean(1 + self.logvar - self.mu**2 - np.exp(self.logvar))

            loss = loss_recon + kl_loss

            # Backward
            grads = self.backward(X, X, acts_enc, zv_enc, acts_dec, zv_dec)
            self.update_parameters_adam(*grads)

            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss={loss:.4f} Recon={loss_recon:.4f} KL={kl_loss:.4f}")