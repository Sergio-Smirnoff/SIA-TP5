import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from autoenconder import BasicAutoencoder, add_salt_and_pepper_noise, add_gaussian_noise

EPOCHS = 25000

def evaluate_accuracy_with_noise(encoder, X_clean, tries=100):
    """
    Calcula el accuracy global sobre todos los caracteres y píxeles,
    usando múltiples tiradas con ruido.
    """
    n_samples, n_features = X_clean.shape
    total_correct = 0

    for i in range(n_samples):
        char_clean = X_clean[i].reshape(1, -1)

        for _ in range(tries):
            if encoder.noise_amount != 0.0:
                noisy = add_salt_and_pepper_noise(char_clean, encoder.noise_amount)
            elif encoder.sigma != 0.0:
                noisy = add_gaussian_noise(char_clean, encoder.sigma)
            else:
                noisy = char_clean.copy()

            pred = encoder.predict(noisy)
            total_correct += np.sum(pred == char_clean)

    # Accuracy global entre 0 y 1
    return total_correct / (n_samples * n_features * tries)


# ---------------------------------------------------------
# ACCURACY POR CARÁCTER
# ---------------------------------------------------------

def evaluate_accuracy_per_character(encoder, X_clean, tries=100):
    """
    Devuelve accuracy por carácter (vector de tamaño N).
    """
    n_samples, n_features = X_clean.shape
    accuracies = np.zeros(n_samples)

    for i in range(n_samples):
        correct = 0
        char_clean = X_clean[i].reshape(1, -1)

        for _ in range(tries):
            if encoder.noise_amount != 0.0:
                noisy = add_salt_and_pepper_noise(char_clean, encoder.noise_amount)
            elif encoder.sigma != 0.0:
                noisy = add_gaussian_noise(char_clean, encoder.sigma)
            else:
                noisy = char_clean.copy()

            pred = encoder.predict(noisy)
            correct += np.sum(pred == char_clean)

        accuracies[i] = correct / (n_features * tries)

    return accuracies


# ---------------------------------------------------------
# ENTRENAR + EVALUAR UNA ARQUITECTURA COMPLETA
# ---------------------------------------------------------

def evaluate_architecture(architecture, X_clean,
                          learning_rate=0.001,
                          epsilon=1e-4,
                          optimizer='adam',
                          activation='tanh',
                          noise_amount=0.0,
                          sigma=0.0,
                          seed=42,
                          test_tries=100):

    encoder = BasicAutoencoder(
        architecture,
        learning_rate=learning_rate,
        epsilon=epsilon,
        optimizer=optimizer,
        activation_function=activation,
        noise_amount=noise_amount,
        sigma=sigma,
        seed=seed
    )

    losses = encoder.train(X_clean, epochs=EPOCHS)

    accuracy_global = evaluate_accuracy_with_noise(
        encoder=encoder,
        X_clean=X_clean,
        tries=test_tries
    )

    acc_per_char = evaluate_accuracy_per_character(
        encoder=encoder,
        X_clean=X_clean,
        tries=test_tries
    )

    return {
        "architecture": architecture,
        "encoder": encoder,
        "losses": np.array(losses),
        "accuracy_global": accuracy_global,
        "accuracy_per_character": acc_per_char
    }


# ---------------------------------------------------------
# HEATMAP POR ARQUITECTURA
# ---------------------------------------------------------

def plot_accuracy_heatmap(accuracies, labels, arch_name, output_path, n_cols=8):
    """
    accuracies: vector con accuracy por carácter
    labels: array de caracteres
    n_cols: número de columnas de la grilla
    """
    n_chars = len(accuracies)
    n_rows = int(np.ceil(n_chars / n_cols))
    
    padded_size = n_rows * n_cols
    padded_accuracies = np.full(padded_size, np.nan)
    padded_accuracies[:n_chars] = accuracies

    padded_labels = np.full(padded_size, "")
    padded_labels[:n_chars] = labels

    data_grid = padded_accuracies.reshape(n_rows, n_cols)
    labels_grid = padded_labels.reshape(n_rows, n_cols)

    plt.figure(figsize=(n_cols*1.5, n_rows*1.5))
    sns.heatmap(
        data_grid,
        annot=labels_grid,
        fmt='',
        cmap="RdPu",
        cbar=True,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray',
        square=True
    )
    plt.title(f"Tasa de acierto por carácter — Arquitectura {arch_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# PLOT LOSS / ERROR POR EPOCH PARA TODAS LAS ARQUITECTURAS
# ---------------------------------------------------------
def plot_loss_per_epoch(results, output_path="outputs/loss_per_epoch.png"):
    plt.figure(figsize=(10,6))
    
    for res in results:
        arch_name = str(res["architecture"])
        plt.plot(res["losses"], label=arch_name)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss por Epoch — Todas las Arquitecturas")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Loss per epoch plot saved at {output_path}")

# ---------------------------------------------------------
# PLOT ACCURACY GLOBAL POR ARQUITECTURA
# ---------------------------------------------------------
def plot_global_accuracy(results, output_path="outputs/global_accuracy.png"):
    arch_names = [str(res["architecture"]) for res in results]
    accuracies = [res["accuracy_global"] for res in results]
    
    plt.figure(figsize=(10,6))
    plt.bar(arch_names, accuracies, color="skyblue", edgecolor="k")
    plt.ylim(0, 1)
    plt.xlabel("Arquitectura")
    plt.ylabel("Accuracy Global")
    plt.title("Tasa de Aciertos Total por Arquitectura")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Global accuracy plot saved at {output_path}")

# ---------------------------------------------------------
# PLOT ACCURACY GLOBAL POR ARQUITECTURA GENERAL DE RUIDOS
# ---------------------------------------------------------
def plot_global_accuracy_all_noises(all_results, architectures, sp_levels, g_levels, output_path="outputs/global_accuracy_all_noises.png"):
    """
    Genera un gráfico de barras agrupadas:
    - Eje X: arquitecturas
    - Barras dentro de cada grupo: cada nivel de ruido (SP + Gaussian)
    """

    noise_labels = ["Clean"]
    for s in sp_levels:
        noise_labels.append(f"SP {int(s*100)}%")
    for g in g_levels:
        noise_labels.append(f"G σ={g}")

    n_arch = len(architectures)
    n_noises = len(noise_labels)

    accuracies = np.zeros((n_arch, n_noises))

    clean_results = all_results[0]
    for i, res in enumerate(clean_results):
        accuracies[i, 0] = res["accuracy_global"]

    col = 1
    for idx_sp in range(len(sp_levels)):
        sp_results = all_results[1 + idx_sp]
        for i, res in enumerate(sp_results):
            accuracies[i, col] = res["accuracy_global"]
        col += 1

    for idx_g in range(len(g_levels)):
        g_results = all_results[1 + len(sp_levels) + idx_g]
        for i, res in enumerate(g_results):
            accuracies[i, col] = res["accuracy_global"]
        col += 1

    # ----- PLOT -----
    x = np.arange(n_arch)
    width = 0.8 / n_noises

    plt.figure(figsize=(14, 7))

    for j in range(n_noises):
        plt.bar(x + j * width, accuracies[:, j], width=width, label=noise_labels[j])

    plt.xticks(x + width * n_noises / 2, [str(a) for a in architectures])

    min_val = np.min(accuracies)
    max_val = np.max(accuracies)

    margin = (max_val - min_val) * 0.2

    if margin == 0:
        margin = 0.01

    plt.ylim(min_val - margin, max_val + margin)

    plt.ylabel("Tasa de Aciertos")
    plt.xlabel("Arquitectura")
    plt.title("Comparación Global de Tasa de Aciertos")
    plt.legend(title="Ruido", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"\nGráfico global de accuracy con todos los ruidos guardado en: {output_path}")



# ---------------------------------------------------------
# FUNCIÓN PRINCIPAL PARA COMPARAR MUCHAS ARQUITECTURAS
# ---------------------------------------------------------

def compare_archs(architectures,
                  X_clean,
                  labels,
                  noise_amount=0.0,
                  sigma=0.0,
                  test_tries=100,
                  save_path="outputs/experiments_results.pkl"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(save_path), "arch_heatmaps"), exist_ok=True)

    results = []

    for arch in architectures:
        print(f"\nEntrenando arquitectura: {arch}")

        res = evaluate_architecture(
            architecture=arch,
            X_clean=X_clean,
            noise_amount=noise_amount,
            sigma=sigma,
            test_tries=test_tries
        )

        results.append(res)

        # --- HEATMAP ---
        heatmap_path = os.path.join(
            os.path.dirname(save_path),
            "arch_heatmaps",
            f"heatmap_{str(arch)}_sp{int(noise_amount*100)}_g{int(sigma*100)}.png"
        )
        plot_accuracy_heatmap(
            res["accuracy_per_character"],
            labels,
            arch_name=str(arch),
            output_path=heatmap_path
        )
        print(f"Heatmap guardado en {heatmap_path}")

    # --- Loss / Accuracy global ---
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    plot_loss_per_epoch(results, output_path=os.path.join(os.path.dirname(save_path), f"{base_name}_loss_per_epoch.png"))
    plot_global_accuracy(results, output_path=os.path.join(os.path.dirname(save_path), f"{base_name}_global_accuracy.png"))
    
    # --- Guardamos resultados ---
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nTodos los experimentos guardados en: {save_path}")
    return results


def run_noise_experiments(X_clean, labels, architectures,
                          salt_pepper_levels=[0.05,0.1,0.15,0.2],
                          gaussian_levels=[0.1,0.2,0.3,0.4,0.5,0.6],
                          test_tries=100):
    """
    Wrapper para correr todos los experimentos:
    - Para cada nivel de Salt&Pepper
    - Para cada nivel de Gaussian
    Genera heatmaps, plots de loss y accuracy global,
    y guarda resultados pickle con nombres diferenciados.
    """
    os.makedirs("outputs", exist_ok=True)

    all_results = []

    # --- CLEAN ---
    res_clean = compare_archs(
        architectures,
        X_clean,
        labels,
        noise_amount=0.0,
        sigma=0.0,
        test_tries=test_tries,
        save_path="outputs/experiments_clean.pkl"
    )
    all_results.append(res_clean)
    
    # --- Salt & Pepper ---
    for sp in salt_pepper_levels:
        print(f"\n===== Experimento Salt&Pepper {sp*100:.0f}% =====")
        res = compare_archs(
            architectures=architectures,
            X_clean=X_clean,
            labels=labels,
            noise_amount=sp,
            sigma=0.0,
            test_tries=test_tries,
            save_path=f"outputs/experiments_saltpepper_{int(sp*100)}.pkl"
        )
        all_results.append(res)

    # --- Gaussian ---
    for sigma in gaussian_levels:
        print(f"\n===== Experimento Gaussian σ={sigma} =====")
        res = compare_archs(
            architectures=architectures,
            X_clean=X_clean,
            labels=labels,
            noise_amount=0.0,
            sigma=sigma,
            test_tries=test_tries,
            save_path=f"outputs/experiments_gaussian_{int(sigma*10)}.pkl"
        )
        all_results.append(res)

    plot_global_accuracy_all_noises(
        all_results,
        architectures,
        salt_pepper_levels,
        gaussian_levels,
        output_path="outputs/global_accuracy_all_noises.png"
    )
