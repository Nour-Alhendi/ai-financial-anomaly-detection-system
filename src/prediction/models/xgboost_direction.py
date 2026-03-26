import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 5

# up/down thresholds — percentile of future_return per ticker (train data only)
UP_PERCENTILE   = 0.75
DOWN_PERCENTILE = 0.25

FEATURES = [
    'anomaly_score', 'combined_anomaly', 'z_score', 'z_anomaly',
    'if_anomaly', 'ae_error', 'ae_anomaly',
    'returns', 'volatility', 'vol_5', 'vol_20', 'vol_change',
    'rolling_mean', 'rolling_std',
    'momentum_5', 'momentum_10', 'trend_strength', 'rsi',
    'return_lag_1', 'return_lag_2', 'return_lag_3',
    'volume_zscore', 'is_high_volume',
    'spx_return', 'etf_return', 'excess_return', 'relative_return', 'sector_relative',
    'is_market_wide', 'is_sector_wide', 'volatility_ratio',
    'regime_encoded', 'beta',
    'z_score_60', 'z_anomaly_60',
    'var_95', 'es_95', 'es_ratio',
]


def load_data():
    all_dfs = []
    for f in sorted(DATA_DIR.glob("*.parquet")):
        if f.stem.startswith("^"):
            continue
        df = pd.read_parquet(f)
        df["ticker"] = f.stem
        all_dfs.append(df)
    data = pd.concat(all_dfs, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"])
    return data.sort_values(["ticker", "Date"]).reset_index(drop=True)


# Generates up/stable/down labels using per-ticker return percentiles (no lookahead)
def generate_labels(data):
    def _future_return(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        future_returns = []
        for i in range(len(closes)):
            if i + HORIZON < len(closes):
                ret = (closes[i + HORIZON] - closes[i]) / closes[i]
            else:
                ret = np.nan
            future_returns.append(ret)
        df["future_return"] = future_returns
        return df

    data = data.groupby("ticker", group_keys=False).apply(_future_return)
    data = data.dropna(subset=["future_return"])

    # Compute thresholds on training data only — no lookahead bias
    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    up_thresh   = data.loc[train_mask].groupby("ticker")["future_return"].quantile(UP_PERCENTILE)
    down_thresh = data.loc[train_mask].groupby("ticker")["future_return"].quantile(DOWN_PERCENTILE)

    data["up_thresh"]   = data["ticker"].map(up_thresh)
    data["down_thresh"] = data["ticker"].map(down_thresh)

    def _label(row):
        if row["future_return"] >= row["up_thresh"]:
            return "up"
        elif row["future_return"] <= row["down_thresh"]:
            return "down"
        else:
            return "stable"

    data["direction"] = data.apply(_label, axis=1)
    data = data.drop(columns=["up_thresh", "down_thresh"])
    return data


# Train on pre-2024 data only
def train(data):
    bool_cols = data[FEATURES].select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    X = data[FEATURES].apply(lambda col: col.astype(float) if col.dtype.name == 'category' else col).fillna(0)
    y = data["direction"].astype(str)

    # LabelEncoder converts "down"/"stable"/"up" → 0/1/2
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    X_train, y_train = X[train_mask], y_enc[train_mask]

    weights = compute_sample_weight("balanced", y_train)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, sample_weight=weights, verbose=0)

    joblib.dump(model, MODEL_DIR / "xgboost_direction.pkl")
    joblib.dump(le,    MODEL_DIR / "direction_label_encoder.pkl")
    print("Model saved.")
    return model, le


# Predict direction for the latest row of each ticker
def predict(data):
    model = joblib.load(MODEL_DIR / "xgboost_direction.pkl")
    le    = joblib.load(MODEL_DIR / "direction_label_encoder.pkl")

    # Use only the most recent data point per ticker (today's snapshot)
    latest = data.groupby("ticker").last().reset_index()

    bool_cols = latest[FEATURES].select_dtypes(include='bool').columns
    latest[bool_cols] = latest[bool_cols].astype(int)
    X_latest = latest[FEATURES].apply(lambda col: col.astype(float) if col.dtype.name == 'category' else col).fillna(0)

    probs     = model.predict_proba(X_latest)
    classes   = list(le.classes_)
    p_up      = probs[:, classes.index('up')]
    p_stable  = probs[:, classes.index('stable')]
    p_down    = probs[:, classes.index('down')]

    # Pick the class with highest probability
    pred_labels = le.inverse_transform(np.argmax(probs, axis=1))

    return pd.DataFrame({
        'ticker':    latest['ticker'].values,
        'direction': pred_labels,
        'p_up':      p_up,
        'p_stable':  p_stable,
        'p_down':    p_down,
    }).sort_values('p_down', ascending=False)


def run():
    print("Loading data...")
    data = load_data()

    print("Generating labels...")
    data = generate_labels(data)

    print("Training model...")
    train(data)

    print("Predicting...")
    results = predict(data)

    print(results.to_string(index=False))


if __name__ == "__main__":
    run()
