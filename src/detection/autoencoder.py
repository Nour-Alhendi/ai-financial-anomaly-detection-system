import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

INPUT_DIR = Path("data/detection")
OUTPUT_DIR = Path("data/detection")
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

# encoder: 4 features → 2 neurons, decoder: 2 → 4
class Autoencoder(Model):
    def __init__(self):
        super().__init__()
        self.encoder = layers.Dense(2, activation="relu")
        self.decoder = layers.Dense(4, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

# train autoencoder and flag rows with high reconstruction error
def autoencoder(file_path):
    df = pd.read_parquet(file_path)
    features = ["returns", "volatility", "z_score", "rolling_std"]
    X = df[features].dropna()

    # normalize to 0-1 (required for sigmoid output)
    X_scaled = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_scaled.values.astype("float32")

    # train only on statistically normal rows
    normal_mask = df.loc[X.index, "z_anomaly"] == False
    X_train = X[normal_mask]
    X_train_scaled = (X_train - X_train.min()) / (X_train.max() - X_train.min())
    X_train_scaled = X_train_scaled.values.astype("float32")

    # train model - input = output (autoencoder trick)
    model = Autoencoder()
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train_scaled, X_train_scaled, epochs=20, batch_size=32, verbose=0)
    
    # reconstruction error: high error = anomaly
    reconstructed = model.predict(X_scaled, verbose=0)
    errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
    threshold = np.percentile(errors, 95)
    df.loc[X.index, "ae_error"] = errors
    df.loc[X.index, "ae_anomaly"] = errors > threshold
    return df

# loops over all files and saves results
def run_autoencoder():
    for file in INPUT_DIR.glob("*.parquet"):
        df = autoencoder(file)
        df.to_parquet(OUTPUT_DIR / file.name)
        print(f"Saved: {file.name}")
        

# Entry point
if __name__ =="__main__":
    run_autoencoder()


    
