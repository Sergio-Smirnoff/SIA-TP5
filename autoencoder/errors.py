
import logging as log
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log.basicConfig(level=log.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')




def read_errors_from_file(file_path: str) -> pd.DataFrame:
    data = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                info = line.strip().split(";")
                arq = info[0]
                fecha = info[1]
                
                epoch_errors = [float(x) for x in info[2].split(",") if x.strip()]
                
                for epoch, error in enumerate(epoch_errors):
                    data.append({
                        "name": arq,
                        "date": fecha,
                        "epoch": epoch,
                        "error": error
                    })
        
        df = pd.DataFrame(data)
        return df
        
    except FileNotFoundError:
        log.error(f"Error file not found: {file_path}")
        return pd.DataFrame()


def main():

    df_errors = read_errors_from_file("outputs/errors.txt")
    if df_errors.empty:
        log.error("No error data to plot.")
        return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_errors, x="epoch", y="error", hue="name")
    plt.title("Error over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    #plt.yscale("log")
    plt.legend(title="Activation")
    plt.grid(True)
    plt.savefig("outputs/error_over_epochs.png")
    plt.close()



if __name__ == "__main__":
    main()