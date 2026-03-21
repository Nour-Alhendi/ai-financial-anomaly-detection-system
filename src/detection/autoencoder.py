# LSTM Autoencoder for anomaly detection — group-aware training.
# One model per group, trained on calm periods (low volatility).
# Detects anomalous sequences (periods), not single days.
# Columns: ae_error, ae_anomaly

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.features import LSTM_AE_FEATURES

INPUT_DIR = Path("data/features")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 20
LSTM_UNITS = 128
LATENT_DIM = 16
DROPOUT = 0.1
SMOOTH_WINDOW = 5
N_FEATURES = len(LSTM_AE_FEATURES)

GROUPS = {
    "Technology-Stable":   {"tickers": ["AAPL", "MSFT", "GOOG"],             "calm_q": 0.65, "percentile": 3},
    "Technology-Volatile": {"tickers": ["NVDA", "AMD"],                       "calm_q": 0.65, "percentile": 3},
    "AI-Stable":           {"tickers": ["CRM", "SNOW"],                       "calm_q": 0.65, "percentile": 5},
    "AI-Volatile":         {"tickers": ["PLTR", "META", "AI"],                "calm_q": 0.75, "percentile": 3},
    "Consumer-Stable":     {"tickers": ["NKE", "MCD", "SBUX"],                "calm_q": 0.65, "percentile": 5},
    "Consumer-Volatile":   {"tickers": ["TSLA", "AMZN"],                      "calm_q": 0.70, "percentile": 3},
    "Financials":          {"tickers": ["JPM", "BAC", "GS", "MS", "BLK"],     "calm_q": 0.70, "percentile": 4},
    "Healthcare":          {"tickers": ["JNJ", "PFE", "UNH", "ABBV", "MRK"],  "calm_q": 0.75, "percentile": 3},
    "Consumer Staples":    {"tickers": ["PG", "KO", "COST", "WMT", "CL"],     "calm_q": 0.70, "percentile": 3},
    "Energy":              {"tickers": ["XOM", "CVX", "COP", "SLB", "EOG"],   "calm_q": 0.70, "percentile": 3},
    "Industrials":         {"tickers": ["CAT", "HON", "BA", "GE", "RTX"],     "calm_q": 0.75, "percentile": 3},
    "Green Energy":        {"tickers": ["BE", "ENPH", "PLUG", "NEE", "FSLR"], "calm_q": 0.75, "percentile": 3},
}


def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH, N_FEATURES))
    x = LSTM(LSTM_UNITS, activation="relu", return_sequences=False)(inputs)
    x = Dropout(DROPOUT)(x)
    x = Dense(LATENT_DIM, activation="relu")(x)
    x = RepeatVector(SEQUENCE_LENGTH)(x)
    x = LSTM(LSTM_UNITS, activation="relu", return_sequences=True)(x)
    x = Dropout(DROPOUT)(x)
    output = TimeDistributed(Dense(N_FEATURES))(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="mse")
    return model


def run_autoencoder():
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    for group_name, params in GROUPS.items():
        tickers = params["tickers"]
        calm_q = params["calm_q"]
        perc = params["percentile"]

        # Load and normalize each stock individually
        frames = []
        for ticker in tickers:
            file = INPUT_DIR / f"{ticker}.parquet"
            if not file.exists():
                print(f"  Skipping {ticker}: file not found")
                continue
            df = pd.read_parquet(file)
            df = df[df["Date"] >= df["Date"].max() - pd.DateOffset(years=4)].reset_index(drop=True)
            df = df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)  # type: ignore
            df["_original_returns"] = df["returns"].copy()

            scaler = MinMaxScaler()
            df[LSTM_AE_FEATURES] = scaler.fit_transform(df[LSTM_AE_FEATURES])
            df["_ticker"] = ticker
            frames.append(df)

        if not frames:
            print(f"Skipping group {group_name}: no data")
            continue

        # Build training sequences from calm periods
        all_train = []
        all_data = []
        for df in frames:
            scaled = df[LSTM_AE_FEATURES].values
            calm_mask = df["volatility"] < df["volatility"].quantile(calm_q)
            train_scaled = df[calm_mask][LSTM_AE_FEATURES].values
            X_train = np.array([train_scaled[i-SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(train_scaled))])
            X_all = np.array([scaled[i-SEQUENCE_LENGTH:i] for i in range(SEQUENCE_LENGTH, len(scaled))])
            all_train.append(X_train)
            all_data.append(X_all)

        X_train_combined = np.concatenate(all_train)
        model = build_model()
        model.fit(X_train_combined, X_train_combined,
                  epochs=30, batch_size=32,
                  validation_split=0.1,
                  callbacks=[early_stop], verbose=0)
        print(f"[{group_name}] trained on {X_train_combined.shape[0]} sequences from {len(frames)} stocks")

        # Predict and save for each stock
        for i, df in enumerate(frames):
            ticker = df["_ticker"].iloc[0]
            X_all = all_data[i]
            X_pred = model.predict(X_all, verbose=0)
            errors = np.mean(np.abs(X_all - X_pred), axis=(1, 2))
            # EMA smoothing — reacts faster to current data, no future leakage
            errors_smooth = pd.Series(errors).ewm(span=SMOOTH_WINDOW).mean()
            threshold = np.percentile(errors_smooth, 100 - perc)

            orig_df = pd.read_parquet(INPUT_DIR / f"{ticker}.parquet")
            orig_df = orig_df[orig_df["Date"] >= orig_df["Date"].max() - pd.DateOffset(years=4)].reset_index(drop=True)
            orig_df = orig_df.dropna(subset=LSTM_AE_FEATURES).reset_index(drop=True)  # type: ignore

            # Align errors to correct rows (first SEQUENCE_LENGTH rows have no prediction)
            orig_df["ae_error"] = np.nan
            orig_df["ae_anomaly"] = False
            error_index = range(SEQUENCE_LENGTH, SEQUENCE_LENGTH + len(errors_smooth))
            orig_df.loc[list(error_index), "ae_error"] = errors_smooth.values
            orig_df.loc[list(error_index), "ae_anomaly"] = errors_smooth.values > threshold

            orig_df.to_parquet(OUTPUT_DIR / f"{ticker}.parquet")
            print(f"  Saved: {ticker}.parquet  ({orig_df['ae_anomaly'].sum()} anomalies)")


if __name__ == "__main__":
    run_autoencoder()
