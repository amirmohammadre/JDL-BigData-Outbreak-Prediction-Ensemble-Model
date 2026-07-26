"""
Master Thesis: JDL-based Situational Awareness for Disease Outbreak Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

TWITTER_PATH = "twitter.csv"
GOOGLE_PATH = "google.csv"
HOSPITAL_PATH = "hospital.csv"
ENV_PATH = "environment.csv"

DATE_COL = 'date'
TARGET = 'new_cases'

LOOKBACK = 14
TEST_SIZE = 60
TOP_FEATURES = 15
EPOCHS = 50
BATCH_SIZE = 16

plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# ==================== LEVEL 0-1: DATA PREPROCESSING ====================

def fix_dates(df, col=DATE_COL):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df.dropna(subset=[col]).sort_values(col).reset_index(drop=True)

def load_data():
    twitter = pd.read_csv(TWITTER_PATH)
    google = pd.read_csv(GOOGLE_PATH)
    hospital = pd.read_csv(HOSPITAL_PATH)
    env = pd.read_csv(ENV_PATH)
    
    twitter = fix_dates(twitter)
    google = fix_dates(google)
    hospital = fix_dates(hospital)
    env = fix_dates(env, col='ts')
    env = env.rename(columns={'ts': DATE_COL})
    
    google_cols = [c for c in google.columns if c != DATE_COL]
    for c in google_cols:
        if google[c].max() > 0:
            google[c] = google[c] / google[c].max()
    
    return twitter, google, hospital, env

def create_features(df, target=TARGET):
    feature_cols = []
    
    for c in ['Confirmed', 'Deaths', 'Recovered', 'new_cases']:
        if c in df.columns:
            feature_cols.append(c)
    
    for c in ['tweets', 'avg_sentiment', 'pos_ratio']:
        if c in df.columns:
            feature_cols.append(c)
    
    google_cols = [c for c in df.columns if c.startswith('google_') or c in ['coronavirus', 'covid19', 'symptoms']]
    feature_cols.extend(google_cols)
    
    for c in ['temp', 'humidity', 'co', 'smoke']:
        if c in df.columns:
            feature_cols.append(c)
    
    feature_cols = list(dict.fromkeys(feature_cols))
    
    for c in ['new_cases', 'tweets', 'coronavirus', 'symptoms', 'temp', 'humidity']:
        if c in df.columns:
            df[f'{c}_roll7'] = df[c].rolling(7, min_periods=1).mean()
            df[f'{c}_roll14'] = df[c].rolling(14, min_periods=1).mean()
            feature_cols.extend([f'{c}_roll7', f'{c}_roll14'])
    
    for c in ['new_cases', 'coronavirus', 'symptoms', 'avg_sentiment']:
        if c in df.columns:
            df[f'{c}_lag1'] = df[c].shift(1)
            df[f'{c}_lag3'] = df[c].shift(3)
            df[f'{c}_lag7'] = df[c].shift(7)
            feature_cols.extend([f'{c}_lag1', f'{c}_lag3', f'{c}_lag7'])
    
    if 'new_cases' in df.columns:
        df['weekly_growth'] = (df['new_cases'] - df['new_cases'].shift(7)) / (df['new_cases'].shift(7) + 1)
        feature_cols.append('weekly_growth')
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    if target not in df.columns:
        df[target] = df['new_cases'].shift(-1).fillna(df['new_cases'])
        df = df.dropna(subset=[target])
    
    return df, feature_cols

def fuse_sources(tw, gt, hp, env):
    start = max(tw[DATE_COL].min(), gt[DATE_COL].min(), hp[DATE_COL].min(), env[DATE_COL].min())
    end = min(tw[DATE_COL].max(), gt[DATE_COL].max(), hp[DATE_COL].max(), env[DATE_COL].max())
    
    tw = tw[(tw[DATE_COL] >= start) & (tw[DATE_COL] <= end)]
    gt = gt[(gt[DATE_COL] >= start) & (gt[DATE_COL] <= end)]
    hp = hp[(hp[DATE_COL] >= start) & (hp[DATE_COL] <= end)]
    env = env[(env[DATE_COL] >= start) & (env[DATE_COL] <= end)]
    
    df = hp.merge(tw, on=DATE_COL, how='inner')
    df = df.merge(gt, on=DATE_COL, how='inner')
    df = df.merge(env, on=DATE_COL, how='inner')
    
    return df

def select_top_features(df, feature_cols, target=TARGET, n_top=TOP_FEATURES):
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target]
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_top]
    selected = [feature_cols[i] for i in top_idx]
    
    print("\n🏆 Top selected features:")
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
    
    return selected, importances, feature_cols

def prepare_sequences(df, features, target=TARGET, lookback=LOOKBACK):
    data = df.sort_values(DATE_COL)
    X = data[features].values
    y = data[target].values
    
    X_seq, y_seq = [], []
    for t in range(lookback, len(data)):
        X_seq.append(X[t-lookback:t])
        y_seq.append(y[t])
    
    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_models(X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_train_flat, y_train_flat, X_val_flat, y_val_flat):
    
    print("Training BiLSTM...")
    lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = lstm_model.fit(X_train_seq, y_train_seq, 
                   validation_data=(X_val_seq, y_val_seq),
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   callbacks=[early_stop], verbose=0)
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, 
                                  learning_rate=0.05, subsample=0.8,
                                  random_state=42)
    xgb_model.fit(X_train_flat, y_train_flat, 
                  eval_set=[(X_val_flat, y_val_flat)], verbose=False)
    
    print("Training ARIMA...")
    try:
        arima_model = ARIMA(y_train_flat, order=(5, 1, 2)).fit()
    except:
        arima_model = ARIMA(y_train_flat, order=(2, 1, 1)).fit()
    
    return lstm_model, xgb_model, arima_model, history

def calculate_dynamic_weights(errors):
    inv_errors = [1.0 / (e + 1e-8) for e in errors]
    total = sum(inv_errors)
    return [e / total for e in inv_errors]

class AlertSystem:
    def __init__(self):
        self.alerts_history = []
    
    def evaluate_threat(self, growth_rate, confidence):
        if growth_rate < 0.05:
            level = "🟢 GREEN"
            alert_type = "LOW_RISK"
            message = "Normal situation."
            color = "green"
        elif growth_rate < 0.15:
            level = "🟡 YELLOW"
            alert_type = "MEDIUM_RISK"
            message = "Monitoring recommended."
            color = "yellow"
        elif growth_rate < 0.30:
            level = "🟠 ORANGE"
            alert_type = "HIGH_RISK"
            message = "Regional alert! Preparation needed."
            color = "orange"
        else:
            level = "🔴 RED"
            alert_type = "CRITICAL"
            message = "National alert! Immediate action required."
            color = "red"
        
        return {'level': level, 'type': alert_type, 'message': message,
                'growth_rate': growth_rate, 'confidence': confidence, 'color': color}


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_feature_importance(importances, feature_names, top_n=15):
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importances[indices][::-1], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: feature_importance.png")

def plot_model_comparison(y_test, predictions_dict):
    plt.figure(figsize=(14, 7))
    colors = {'BiLSTM': 'blue', 'XGBoost': 'green', 'ARIMA': 'orange', 'Ensemble': 'red'}
    for name, pred in predictions_dict.items():
        plt.plot(pred, '--', label=name, color=colors.get(name, 'gray'), alpha=0.7, linewidth=1.5)
    plt.plot(y_test, 'k-', label='Actual', linewidth=2)
    plt.xlabel('Days')
    plt.ylabel('New Cases')
    plt.title('Model Comparison - Prediction Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: model_comparison.png")

def plot_error_distribution(errors_dict):
    plt.figure(figsize=(10, 6))
    models = list(errors_dict.keys())
    mae_values = [errors_dict[m]['mae'] for m in models]
    rmse_values = [errors_dict[m]['rmse'] for m in models]
    x = np.arange(len(models))
    width = 0.35
    bars1 = plt.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    bars2 = plt.bar(x + width/2, rmse_values, width, label='RMSE', color='coral')
    plt.xlabel('Models')
    plt.ylabel('Error')
    plt.title('Model Error Comparison')
    plt.xticks(x, models)
    plt.legend()
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', ha='center', va='bottom')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: error_distribution.png")

def plot_prediction_vs_actual(y_test, y_pred, model_name="Ensemble"):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, c='steelblue', edgecolors='white', linewidth=0.5)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name}: Prediction vs Actual')
    plt.legend()
    plt.grid(True, alpha=0.3)
    r2 = r2_score(y_test, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(f'prediction_vs_actual_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: prediction_vs_actual_{model_name.lower()}.png")

def plot_residuals(y_test, y_pred, model_name="Ensemble"):
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(residuals, 'o-', alpha=0.5, markersize=3, linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Time (days)')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residuals Over Time')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - Residual Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'residuals_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: residuals_{model_name.lower()}.png")

def plot_learning_curve(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('BiLSTM - Learning Curve (Loss)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('BiLSTM - Learning Curve (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: learning_curve.png")

def plot_correlation_heatmap(df, features):
    plt.figure(figsize=(14, 12))
    corr_features = [f for f in features if f in df.columns][:15]
    corr_matrix = df[corr_features].corr()
    im = plt.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Correlation')
    plt.xticks(range(len(corr_features)), corr_features, rotation=45, ha='right')
    plt.yticks(range(len(corr_features)), corr_features)
    for i in range(len(corr_features)):
        for j in range(len(corr_features)):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center", 
                    color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white", fontsize=8)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: correlation_heatmap.png")

def generate_grafana_alerts(threat):
    import json
    from datetime import datetime
    grafana_alerts = {
        "title": "JDL Disease Outbreak Alert System",
        "generated_at": datetime.now().isoformat(),
        "alerts": [
            {"name": "GREEN - Normal", "condition": "growth_rate < 0.05", "color": "green", "message": "Normal situation."},
            {"name": "YELLOW - Monitoring", "condition": "0.05 <= growth_rate < 0.15", "color": "yellow", "message": "Monitoring recommended."},
            {"name": "ORANGE - Regional Alert", "condition": "0.15 <= growth_rate < 0.30", "color": "orange", "message": "Regional alert!"},
            {"name": "RED - National Alert", "condition": "growth_rate >= 0.30", "color": "red", "message": "NATIONAL ALERT!"}
        ],
        "current_status": {
            "growth_rate": float(threat['growth_rate']),
            "confidence": float(threat['confidence']),
            "level": threat['level'],
            "message": threat['message']
        }
    }
    with open('grafana_alerts.json', 'w', encoding='utf-8') as f:
        json.dump(grafana_alerts, f, indent=2, ensure_ascii=False)
    print("✅ Saved: grafana_alerts.json")


def save_to_sqlite(y_test_seq, y_pred_ensemble, dates, threat, filename="predictions_for_grafana.db"):
    import sqlite3
    
    formatted_dates = []
    for d in dates:
        if isinstance(d, pd.Timestamp):
            formatted_dates.append(d.strftime('%Y-%m-%d %H:%M:%S'))
        elif isinstance(d, str):
            d_clean = d.replace('T', ' ') if 'T' in d else d
            formatted_dates.append(pd.to_datetime(d_clean).strftime('%Y-%m-%d %H:%M:%S'))
        else:
            formatted_dates.append(pd.to_datetime(d).strftime('%Y-%m-%d %H:%M:%S'))
    
    df_sqlite = pd.DataFrame({
        'time': formatted_dates,
        'actual': y_test_seq,
        'predicted': y_pred_ensemble,
        'growth_rate': [threat['growth_rate']] * len(y_test_seq),
        'confidence': [threat['confidence']] * len(y_test_seq),
        'alert_level': [threat['type']] * len(y_test_seq),
        'alert_color': [threat['color']] * len(y_test_seq)
    })
    
    conn = sqlite3.connect(filename)
    
    conn.execute("DROP TABLE IF EXISTS predictions")
    
    conn.execute("""
        CREATE TABLE predictions (
            time TEXT PRIMARY KEY,
            actual REAL,
            predicted REAL,
            growth_rate REAL,
            confidence REAL,
            alert_level TEXT,
            alert_color TEXT
        )
    """)
    
    data_to_insert = df_sqlite.values.tolist()
    conn.executemany("""
        INSERT INTO predictions (time, actual, predicted, growth_rate, confidence, alert_level, alert_color)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, data_to_insert)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_time ON predictions(time)")
    
    conn.commit()
    conn.close()
    
    print(f"✅ Saved SQLite database: {filename}")
    print(f"   - Table name: predictions")
    print(f"   - Total records: {len(df_sqlite)}")
    print(f"   - Date range: {formatted_dates[0]} to {formatted_dates[-1]}")
    print(f"   - Time column format: TEXT (YYYY-MM-DD HH:MM:SS)")
    
    print("\n📊 Sample data (first 5 records):")
    print(df_sqlite.head().to_string(index=False))
    
    return filename


# ==================== MULTI-HORIZON FUNCTIONS (By comparing XGBoost) ====================

def train_and_evaluate_for_horizon(df_features, selected_features, horizon, test_size=60):
    df = df_features.copy()
    
    target_horizon = f'target_h{horizon}'
    df[target_horizon] = df['new_cases'].shift(-horizon)
    df = df.dropna(subset=[target_horizon])
    
    X_seq, y_seq = prepare_sequences(df, selected_features, target=target_horizon)
    
    if len(X_seq) < test_size + 10:
        return None
    
    split = len(X_seq) - test_size
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]
    
    val_split = int(len(X_train_seq) * 0.8)
    X_train_seq, X_val_seq = X_train_seq[:val_split], X_train_seq[val_split:]
    y_train_seq, y_val_seq = y_train_seq[:val_split], y_train_seq[val_split:]
    
    X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
    X_val_flat = X_val_seq.reshape(len(X_val_seq), -1)
    X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    scaler_X.fit(X_train_flat)
    scaler_y.fit(y_train_seq.reshape(-1, 1))
    
    X_train_norm = scaler_X.transform(X_train_flat).reshape(X_train_seq.shape)
    X_val_norm = scaler_X.transform(X_val_flat).reshape(X_val_seq.shape)
    X_test_norm = scaler_X.transform(X_test_flat).reshape(X_test_seq.shape)
    
    y_train_norm = scaler_y.transform(y_train_seq.reshape(-1, 1)).ravel()
    y_val_norm = scaler_y.transform(y_val_seq.reshape(-1, 1)).ravel()
    
    # BiLSTM
    lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    lstm_model.fit(X_train_norm, y_train_norm, 
                   validation_data=(X_val_norm, y_val_norm),
                   epochs=30, batch_size=BATCH_SIZE,
                   callbacks=[early_stop], verbose=0)
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, 
                                  learning_rate=0.05, subsample=0.8,
                                  random_state=42)
    xgb_model.fit(X_train_flat, y_train_seq, 
                  eval_set=[(X_val_flat, y_val_seq)], verbose=False)
    
    # ARIMA
    try:
        arima_model = ARIMA(y_train_seq, order=(5, 1, 2)).fit()
    except:
        arima_model = ARIMA(y_train_seq, order=(2, 1, 1)).fit()
    
    pred_lstm = scaler_y.inverse_transform(lstm_model.predict(X_test_norm, verbose=0)).ravel()
    pred_xgb = xgb_model.predict(X_test_flat)
    
    try:
        pred_arima = arima_model.forecast(steps=len(y_test_seq))
        pred_arima = pred_arima[:len(y_test_seq)]
    except:
        pred_arima = np.ones(len(y_test_seq)) * np.mean(y_train_seq)
    
    errors = [
        mean_absolute_error(y_test_seq, pred_lstm),
        mean_absolute_error(y_test_seq, pred_xgb),
        mean_absolute_error(y_test_seq, pred_arima)
    ]
    weights = calculate_dynamic_weights(errors)
    
    y_pred_ensemble = weights[0] * pred_lstm + weights[1] * pred_xgb + weights[2] * pred_arima
    y_pred_ensemble = np.maximum(y_pred_ensemble, 0)
    
    return {
        'horizon': horizon,
        # Ensemble
        'ensemble_mae': mean_absolute_error(y_test_seq, y_pred_ensemble),
        'ensemble_rmse': mean_squared_error(y_test_seq, y_pred_ensemble, squared=False),
        'ensemble_r2': r2_score(y_test_seq, y_pred_ensemble),
        'ensemble_predictions': y_pred_ensemble,
        # XGBoost
        'xgb_mae': mean_absolute_error(y_test_seq, pred_xgb),
        'xgb_rmse': mean_squared_error(y_test_seq, pred_xgb, squared=False),
        'xgb_r2': r2_score(y_test_seq, pred_xgb),
        'xgb_predictions': pred_xgb,
        # Actual
        'actual': y_test_seq
    }


def evaluate_multi_horizons(df_features, selected_features, horizons=[1, 3, 7, 14]):
    print("\n" + "="*70)
    print("MULTI-HORIZON EVALUATION (Comparison of Ensemble and XGBoost)")
    print("="*70)
    
    results = []
    for h in horizons:
        print(f"\n🔄 Training for {h}-day ahead prediction...")
        result = train_and_evaluate_for_horizon(df_features, selected_features, h)
        if result:
            results.append(result)
            print(f"   ✅ Ensemble - MAE: {result['ensemble_mae']:.2f}, R²: {result['ensemble_r2']:.3f}")
            print(f"   ✅ XGBoost  - MAE: {result['xgb_mae']:.2f}, R²: {result['xgb_r2']:.3f}")
        else:
            print(f"   ⚠️ Not enough data for {h}-day horizon")
    return results


def plot_multi_horizon_comparison(results):
    if not results:
        print("No results to plot")
        return
    
    horizons = [r['horizon'] for r in results]
    ensemble_mae = [r['ensemble_mae'] for r in results]
    xgb_mae = [r['xgb_mae'] for r in results]
    ensemble_r2 = [r['ensemble_r2'] for r in results]
    xgb_r2 = [r['xgb_r2'] for r in results]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    x = np.arange(len(horizons))
    width = 0.35
    bars1 = plt.bar(x - width/2, ensemble_mae, width, label='Ensemble', color='steelblue')
    bars2 = plt.bar(x + width/2, xgb_mae, width, label='XGBoost', color='coral')
    plt.xlabel('Prediction Horizon (days)')
    plt.ylabel('MAE')
    plt.title('MAE Comparison: Ensemble vs XGBoost')
    plt.xticks(x, [f'{h}-day' for h in horizons])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(1, 2, 2)
    bars3 = plt.bar(x - width/2, ensemble_r2, width, label='Ensemble', color='steelblue')
    bars4 = plt.bar(x + width/2, xgb_r2, width, label='XGBoost', color='coral')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.xlabel('Prediction Horizon (days)')
    plt.ylabel('R² Score')
    plt.title('R² Comparison: Ensemble vs XGBoost')
    plt.xticks(x, [f'{h}-day' for h in horizons])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ensemble_vs_xgboost_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: ensemble_vs_xgboost_comparison.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i >= 4:
            break
        ax = axes[i]
        horizon = result['horizon']
        y_actual = result['actual']
        y_pred_ensemble = result['ensemble_predictions']
        y_pred_xgb = result['xgb_predictions']
        
        ax.plot(y_actual, 'b-', label='Actual', linewidth=1.5)
        ax.plot(y_pred_ensemble, 'r--', label='Ensemble', linewidth=1.5)
        ax.plot(y_pred_xgb, 'g--', label='XGBoost', linewidth=1.5)
        ax.set_title(f'{horizon}-Day Ahead (Ens R²={result["ensemble_r2"]:.3f}, XGB R²={result["xgb_r2"]:.3f})')
        ax.set_xlabel('Days')
        ax.set_ylabel('New Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_horizon_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: multi_horizon_predictions_comparison.png")
    
    print("\n" + "="*70)
    print("Comparison table of Ensemble and XGBoost performance across different horizons")
    print("="*70)
    print(f"{'Horizon':<12} {'Ensemble MAE':<15} {'XGBoost MAE':<15} {'Ensemble R²':<15} {'XGBoost R²':<15}")
    print("-"*70)
    for r in results:
        print(f"{r['horizon']}-day{'':<6} {r['ensemble_mae']:<15.2f} {r['xgb_mae']:<15.2f} {r['ensemble_r2']:<15.3f} {r['xgb_r2']:<15.3f}")
    
    pd.DataFrame(results).to_csv('ensemble_vs_xgboost_results.csv', index=False)
    print("\n✅ Saved: ensemble_vs_xgboost_results.csv")


def plot_multi_horizon_results_only(results):
    if not results:
        return
    
    horizons = [r['horizon'] for r in results]
    ensemble_mae = [r['ensemble_mae'] for r in results]
    ensemble_r2 = [r['ensemble_r2'] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(horizons, ensemble_mae, 'o-', color='steelblue', linewidth=2, markersize=8)
    plt.xlabel('Prediction Horizon (days)')
    plt.ylabel('MAE')
    plt.title('MAE vs Prediction Horizon (Ensemble)')
    plt.grid(True, alpha=0.3)
    for h, m in zip(horizons, ensemble_mae):
        plt.annotate(f'{m:.1f}', (h, m), textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.subplot(1, 2, 2)
    colors = ['green' if r2 > 0.85 else 'orange' if r2 > 0.7 else 'red' for r2 in ensemble_r2]
    bars = plt.bar([f'{h}-day' for h in horizons], ensemble_r2, color=colors, edgecolor='black')
    plt.axhline(y=0.85, color='green', linestyle='--', alpha=0.7, label='Good (0.85)')
    plt.axhline(y=0.70, color='orange', linestyle='--', alpha=0.7, label='Acceptable (0.70)')
    plt.xlabel('Prediction Horizon')
    plt.ylabel('R² Score')
    plt.title('R² Score Degradation (Ensemble)')
    plt.ylim(0.5, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    for bar, r2 in zip(bars, ensemble_r2):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r2:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('multi_horizon_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: multi_horizon_results.png")


def plot_multi_horizon_predictions(results):
    if not results:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i >= 4:
            break
        ax = axes[i]
        horizon = result['horizon']
        y_actual = result['actual']
        y_pred = result['ensemble_predictions']
        
        ax.plot(y_actual, 'b-', label='Actual', linewidth=1.5)
        ax.plot(y_pred, 'r--', label='Ensemble Prediction', linewidth=1.5)
        ax.fill_between(range(len(y_actual)), 
                       y_pred - result['ensemble_rmse'], 
                       y_pred + result['ensemble_rmse'], 
                       alpha=0.2, color='red')
        ax.set_title(f'{horizon}-Day Ahead (R²={result["ensemble_r2"]:.3f})')
        ax.set_xlabel('Days')
        ax.set_ylabel('New Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_horizon_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: multi_horizon_predictions.png")


# ==================== MAIN ====================
def main():
    print("="*70)
    print("JDL-BASED SITUATIONAL AWARENESS FOR DISEASE OUTBREAK PREDICTION")
    print("="*70)
    
    print("\n[Level 0-1] Loading data...")
    twitter, google, hospital, env = load_data()
    print(f"  Twitter: {len(twitter)}, Google: {len(google)}, Hospital: {len(hospital)}, Environment: {len(env)}")
    
    print("\n[Level 2] Fusing data...")
    fused = fuse_sources(twitter, google, hospital, env)
    print(f"  Fused data: {len(fused)} records")
    
    print("\n[Level 2] Feature engineering...")
    featured, feature_cols = create_features(fused)
    print(f"  Features created: {len(feature_cols)}")
    
    print("\n[Level 3] Feature selection...")
    selected_features, importances, all_features = select_top_features(featured, feature_cols)
    plot_feature_importance(importances, all_features, top_n=15)
    
    print("\n[Level 3] Preparing sequences...")
    X_seq, y_seq = prepare_sequences(featured, selected_features)
    
    if len(X_seq) < TEST_SIZE + 10:
        print(f"Error: Not enough data. Need at least {TEST_SIZE+10} sequences, got {len(X_seq)}")
        return
    
    split = len(X_seq) - TEST_SIZE
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]
    
    val_split = int(len(X_train_seq) * 0.8)
    X_train_seq, X_val_seq = X_train_seq[:val_split], X_train_seq[val_split:]
    y_train_seq, y_val_seq = y_train_seq[:val_split], y_train_seq[val_split:]
    
    X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
    X_val_flat = X_val_seq.reshape(len(X_val_seq), -1)
    X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)
    
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_train_flat)
    scaler_y.fit(y_train_seq.reshape(-1, 1))
    
    X_train_norm = scaler_X.transform(X_train_flat).reshape(X_train_seq.shape)
    X_val_norm = scaler_X.transform(X_val_flat).reshape(X_val_seq.shape)
    X_test_norm = scaler_X.transform(X_test_flat).reshape(X_test_seq.shape)
    
    y_train_norm = scaler_y.transform(y_train_seq.reshape(-1, 1)).ravel()
    y_val_norm = scaler_y.transform(y_val_seq.reshape(-1, 1)).ravel()
    
    print("\n[Level 3] Training models...")
    lstm_model, xgb_model, arima_model, history = train_models(
        X_train_norm, y_train_norm, X_val_norm, y_val_norm,
        X_train_flat, y_train_seq, X_val_flat, y_val_seq
    )
    
    plot_learning_curve(history)
    
    print("\n[Level 3] Making predictions...")
    pred_lstm = scaler_y.inverse_transform(lstm_model.predict(X_test_norm, verbose=0)).ravel()
    pred_xgb = xgb_model.predict(X_test_flat)
    
    try:
        pred_arima = arima_model.forecast(steps=len(y_test_seq))
        pred_arima = pred_arima[:len(y_test_seq)]
    except:
        pred_arima = np.ones(len(y_test_seq)) * np.mean(y_train_seq)
    
    errors_list = [
        mean_absolute_error(y_test_seq[:len(pred_lstm)], pred_lstm),
        mean_absolute_error(y_test_seq, pred_xgb),
        mean_absolute_error(y_test_seq, pred_arima)
    ]
    weights = calculate_dynamic_weights(errors_list)
    
    y_pred_ensemble = weights[0] * pred_lstm + weights[1] * pred_xgb + weights[2] * pred_arima
    y_pred_ensemble = np.maximum(y_pred_ensemble, 0)
    
    mae = mean_absolute_error(y_test_seq, y_pred_ensemble)
    rmse = mean_squared_error(y_test_seq, y_pred_ensemble, squared=False)
    r2 = r2_score(y_test_seq, y_pred_ensemble)
    
    print("\n" + "="*60)
    print("📈 Evaluation Results (1-day ahead)")
    print("="*60)
    print(f"Ensemble Model: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f}")
    print(f"XGBoost alone: MAE={mean_absolute_error(y_test_seq, pred_xgb):.2f}, R²={r2_score(y_test_seq, pred_xgb):.3f}")
    print(f"Weights - BiLSTM: {weights[0]:.3f}, XGBoost: {weights[1]:.3f}, ARIMA: {weights[2]:.3f}")
    
    errors_dict = {
        'BiLSTM': {'mae': mean_absolute_error(y_test_seq[:len(pred_lstm)], pred_lstm),
                   'rmse': mean_squared_error(y_test_seq[:len(pred_lstm)], pred_lstm, squared=False)},
        'XGBoost': {'mae': mean_absolute_error(y_test_seq, pred_xgb),
                    'rmse': mean_squared_error(y_test_seq, pred_xgb, squared=False)},
        'ARIMA': {'mae': mean_absolute_error(y_test_seq, pred_arima),
                  'rmse': mean_squared_error(y_test_seq, pred_arima, squared=False)},
        'Ensemble': {'mae': mae, 'rmse': rmse}
    }
    
    predictions_dict = {
        'BiLSTM': pred_lstm[:len(y_test_seq)],
        'XGBoost': pred_xgb,
        'ARIMA': pred_arima,
        'Ensemble': y_pred_ensemble
    }
    
    plot_model_comparison(y_test_seq, predictions_dict)
    plot_error_distribution(errors_dict)
    plot_prediction_vs_actual(y_test_seq, y_pred_ensemble, "Ensemble")
    plot_residuals(y_test_seq, y_pred_ensemble, "Ensemble")
    plot_correlation_heatmap(featured, selected_features[:15])
    
    # ==================== MULTI-HORIZON EVALUATION ====================
    print("\n" + "="*70)
    print("LEVEL 3 EXTENDED: Multi-Horizon Prediction Evaluation")
    print("="*70)
    
    multi_results = evaluate_multi_horizons(featured, selected_features, horizons=[1, 3, 7, 14])
    
    if multi_results:
        plot_multi_horizon_comparison(multi_results)
        plot_multi_horizon_results_only(multi_results)
        plot_multi_horizon_predictions(multi_results)
    else:
        print("⚠️ Multi-horizon evaluation skipped (not enough data)")
    
    last_week_avg = np.mean(y_test_seq[-7:]) if len(y_test_seq) >= 7 else np.mean(y_test_seq)
    growth = (y_pred_ensemble[-1] - last_week_avg) / (last_week_avg + 1)
    growth = max(0, min(1, growth))
    confidence = 1.0 / (1.0 + mae / (np.mean(y_test_seq) + 1))
    
    alert = AlertSystem()
    threat = alert.evaluate_threat(growth, confidence)
    
    print(f"\n🚨 Alert: {threat['type']}")
    print(f"   Level: {threat['level']}")
    print(f"   Growth Rate: {growth*100:.1f}%")
    print(f"   Confidence: {confidence*100:.1f}%")
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_seq, 'b-', label='Actual Cases', linewidth=2)
    plt.plot(y_pred_ensemble, 'r--', label='Ensemble Prediction', linewidth=2)
    plt.fill_between(range(len(y_test_seq)), y_pred_ensemble - rmse, y_pred_ensemble + rmse, 
                     alpha=0.2, color='red', label='Uncertainty (±RMSE)')
    plt.xlabel('Days')
    plt.ylabel('Number of New Cases')
    plt.title(f'Disease Outbreak Prediction - JDL Model (R² = {r2:.3f}, MAE = {mae:.1f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('final_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Saved: final_prediction.png")

    
    
    data_sorted = featured.sort_values(DATE_COL)
    dates_all = data_sorted[DATE_COL].values
    
    if len(dates_all) >= len(y_test_seq):
        test_dates = dates_all[-len(y_test_seq):]
    else:
        test_dates = pd.date_range(start='2020-01-01', periods=len(y_test_seq), freq='D')
    
  
        
    save_to_sqlite(y_test_seq, y_pred_ensemble, test_dates, threat)

    generate_grafana_alerts(threat)
    
    print("\n✅ ALL COMPLETED")


if __name__ == "__main__":
    main()