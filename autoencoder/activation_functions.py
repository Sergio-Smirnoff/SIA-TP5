import numpy as np

def sigmoid(x):
    """Función sigmoide: f(x) = 1 / (1 + e^-x)"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para estabilidad

def sigmoid_derivative(x):
    """Derivada de sigmoide: f'(x) = f(x) * (1 - f(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU: f(x) = max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivada de ReLU: f'(x) = 1 si x > 0, sino 0"""
    return (x > 0).astype(float)

def tanh(x):
    """Tangente hiperbólica"""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivada de tanh: f'(x) = 1 - tanh²(x)"""
    return 1 - np.tanh(x) ** 2

