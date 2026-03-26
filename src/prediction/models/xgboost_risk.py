import pandas as pd
from pathlib import Path
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

ROOT      = Path(__file__).resolve().parents[3]
DATA_DIR  = ROOT / "data/detection"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
HORIZON   = 5

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

# load data
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

# Generates high/low risk labels using per-ticker VaR (no lookahead)
def generate_labels(data):
    def _drawdown(df):
        df = df.copy().sort_values("Date")
        closes = df["Close"].values
        future_drawdowns = []
        for i in range(len(closes)):
            window = closes[i:i + HORIZON]
            drawdown = abs((window.min() - closes[i]) / closes[i])
            future_drawdowns.append(drawdown)
        df["future_max_drawdown"] = future_drawdowns
        return df

    data = data.groupby("ticker", group_keys=False).apply(_drawdown)
    data = data.dropna(subset=["future_max_drawdown"])

    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    var_by_ticker = data.loc[train_mask].groupby("ticker")["future_max_drawdown"].quantile(0.85)

    data["_drawdown_thresh"] = data["ticker"].map(var_by_ticker)
    data["risk_level"] = (data["future_max_drawdown"] >= data["_drawdown_thresh"]).map({True: "high", False: "low"})
    data = data.drop(columns=["_drawdown_thresh"])
    return data


# Train on pre-2024 data only — scale_pos_weight corrects for class imbalance (high is rare ~15%)
def train(data):
    bool_cols = data[FEATURES].select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    X = data[FEATURES].apply(lambda col: col.astype(float) if col.dtype.name == 'category' else col).fillna(0)
    y = data["risk_level"].astype(str)

    # LabelEncoder converts "high"/"low" → 0/1 so XGBoost can work with numeric labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    train_mask = data["Date"] < pd.Timestamp("2024-01-01")
    X_train, y_train = X[train_mask], y_enc[train_mask]

    # scale_pos_weight = neg/pos tells XGBoost how much rarer "high" is vs "low"
    # without this, the model would almost always predict "low" and ignore "high"
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=neg/pos,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, verbose=0)

    joblib.dump(model, MODEL_DIR / "xgboost_risk.pkl")
    joblib.dump(le,    MODEL_DIR / "risk_label_encoder.pkl")
    print("Model saved.")
    return model, le


# Predict risk level for the latest row of each ticker
def predict(data):
    model = joblib.load(MODEL_DIR / "xgboost_risk.pkl")
    le    = joblib.load(MODEL_DIR / "risk_label_encoder.pkl")

    # Use only the most recent data point per ticker (today's snapshot)
    latest = data.groupby("ticker").last().reset_index()

    bool_cols = latest[FEATURES].select_dtypes(include='bool').columns
    latest[bool_cols] = latest[bool_cols].astype(int)
    X_latest = latest[FEATURES].apply(lambda col: col.astype(float) if col.dtype.name == 'category' else col).fillna(0)

    probs  = model.predict_proba(X_latest)
    p_high = probs[:, list(le.classes_).index('high')]
    # Apply threshold: if probability of "high" >= 0.5 → classify as high risk
    pred_labels = np.where(p_high >= THRESHOLD, 'high', 'low')

    return pd.DataFrame({
        'ticker':        latest['ticker'].values,
        'risk_level':    pred_labels,
        'p_low':         probs[:, list(le.classes_).index('low')],
        'p_high':        p_high,
        'anomaly_score': latest['anomaly_score'].values,
    }).sort_values('p_high', ascending=False)


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
