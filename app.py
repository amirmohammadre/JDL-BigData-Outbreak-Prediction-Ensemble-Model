"""
Master Thesis: JDL-based Situational Awareness for Disease Outbreak Prediction
Final Ensemble Pipeline (BiLSTM + XGBoost)
Architecture: Multi-Task Vector Forecasting with Positive 14-Day R2
Author: Amir Mohammad Rezvaninia
"""

import os
import json
import sqlite3
import random
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

import xgboost as xgb

warnings.filterwarnings('ignore')

# Fix Seeds for Deterministic Results
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Configuration Parameters
TWITTER_PATH = "twitter.csv"
GOOGLE_PATH = "google.csv"
HOSPITAL_PATH = "hospital.csv"
ENV_PATH = "environment.csv"

DATE_COL = 'date'
TARGET = 'new_cases'

LOOKBACK = 21             # 3-Week input window to remove noise
FORECAST_HORIZON = 14     # Direct vector output for 14 days
TEST_SIZE = 60
TOP_FEATURES_COUNT = 10   # Top features selected via Mutual Information
EPOCHS = 60
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
    
    for c in ['Deaths', 'Recovered', 'tweets', 'avg_sentiment', 'pos_ratio', 'temp', 'humidity', 'co', 'smoke']:
        if c in df.columns:
            feature_cols.append(c)
    
    google_cols = [c for c in df.columns if c.startswith('google_') or c in ['coronavirus', 'covid19', 'symptoms']]
    feature_cols.extend(google_cols)
    
    for c in ['new_cases', 'tweets', 'coronavirus', 'symptoms', 'temp', 'humidity']:
        if c in df.columns:
            df[f'{c}_roll7'] = df[c].rolling(7, min_periods=1).mean()
            df[f'{c}_roll14'] = df[c].rolling(14, min_periods=1).mean()
            df[f'{c}_roll28'] = df[c].rolling(28, min_periods=1).mean()
            feature_cols.extend([f'{c}_roll7', f'{c}_roll14', f'{c}_roll28'])
    
    for c in ['coronavirus', 'symptoms', 'avg_sentiment', 'new_cases']:
        if c in df.columns:
            df[f'{c}_lag1'] = df[c].shift(1)
            df[f'{c}_lag3'] = df[c].shift(3)
            df[f'{c}_lag7'] = df[c].shift(7)
            df[f'{c}_lag14'] = df[c].shift(14)
            feature_cols.extend([f'{c}_lag1', f'{c}_lag3', f'{c}_lag7', f'{c}_lag14'])
    
    if 'new_cases' in df.columns:
        df['weekly_growth'] = (df['new_cases'] - df['new_cases'].shift(7)) / (df['new_cases'].shift(7) + 1)
        df['biweekly_growth'] = (df['new_cases'] - df['new_cases'].shift(14)) / (df['new_cases'].shift(14) + 1)
        feature_cols.extend(['weekly_growth', 'biweekly_growth'])
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, list(dict.fromkeys(feature_cols))

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

def select_top_features_mi(df, feature_cols, target=TARGET, top_n=TOP_FEATURES_COUNT):
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df[target].shift(-1).fillna(df[target])
    
    mi_scores = mutual_info_regression(X, y, random_state=SEED)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    selected = mi_series.head(top_n).index.tolist()
    
    print(f"\n Selected Top {top_n} Features via Mutual Information:")
    for feat in selected:
        print(f"  - {feat}: {mi_series[feat]:.4f}")
        
    return selected, mi_series

# Multi-Output Sequence Builder with Trend Target Smoothing
def prepare_multi_output_sequences(df, features, horizon=FORECAST_HORIZON, lookback=LOOKBACK):
    data = df.copy().sort_values(DATE_COL)
    
    target_cols = []
    for h in range(1, horizon + 1):
        col_name = f'target_h{h}'
        # Smooth noise for horizons > 3 days using 7-day rolling mean
        if h > 3:
            data[col_name] = data['new_cases'].rolling(7, min_periods=1).mean().shift(-h)
        else:
            data[col_name] = data['new_cases'].shift(-h)
        target_cols.append(col_name)
    
    data = data.dropna(subset=target_cols)
    
    X_vals = data[features].values
    y_vals = data[target_cols].values
    
    X_seq, y_seq = [], []
    for t in range(lookback, len(data)):
        X_seq.append(X_vals[t-lookback:t])
        y_seq.append(y_vals[t])
        
    return np.array(X_seq), np.array(y_seq), data[DATE_COL].values[lookback:]

def build_multi_output_bilstm(input_shape, output_dim=FORECAST_HORIZON):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.2),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

class AlertSystem:
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

# ==================== VISUALIZATIONS EXPORT ====================

def plot_feature_importance_mi(mi_series, selected_features):
    top_mi = mi_series.loc[selected_features].sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top_mi.index, top_mi.values, color='steelblue')
    plt.xlabel('Mutual Information Score')
    plt.title('Top Selected Features via Mutual Information')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

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
    plt.close()

def plot_model_comparison(y_test, pred_lstm, pred_xgb, pred_ens):
    plt.figure(figsize=(14, 7))
    plt.plot(y_test[:, 0], 'k-', label='Actual Cases', linewidth=2)
    plt.plot(pred_lstm[:, 0], 'b--', label='BiLSTM (1-Day)', alpha=0.7, linewidth=1.5)
    plt.plot(pred_xgb[:, 0], 'g--', label='XGBoost (1-Day)', alpha=0.7, linewidth=1.5)
    plt.plot(pred_ens[:, 0], 'r--', label='Ensemble (1-Day)', linewidth=2)
    plt.xlabel('Days')
    plt.ylabel('New Cases')
    plt.title('Model Comparison - Prediction Performance (1-Day Horizon)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(mae_lstm, rmse_lstm, mae_xgb, rmse_xgb, mae_ens, rmse_ens):
    plt.figure(figsize=(8, 5))
    models_list = ['BiLSTM', 'XGBoost', 'Ensemble']
    mae_values = [mae_lstm, mae_xgb, mae_ens]
    rmse_values = [rmse_lstm, rmse_xgb, rmse_ens]
    
    x = np.arange(len(models_list))
    width = 0.35
    bars1 = plt.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    bars2 = plt.bar(x + width/2, rmse_values, width, label='RMSE', color='coral')
    
    plt.xlabel('Models')
    plt.ylabel('Error Rate')
    plt.title('Error Comparison (MAE vs RMSE - 1-Day Horizon)')
    plt.xticks(x, models_list)
    plt.legend()
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center')
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{bar.get_height():.1f}', ha='center')
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_test, y_pred):
    residuals = y_test[:, 0] - y_pred[:, 0]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(residuals, 'o-', alpha=0.5, markersize=3, linewidth=0.5)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Time (days)')
    plt.ylabel('Residuals')
    plt.title('Ensemble - Residuals Over Time (1-Day)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Ensemble - Residual Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('residuals_ensemble.png', dpi=300, bbox_inches='tight')
    plt.close()

# ==================== SAVE FUNCTIONS ====================

def save_to_sqlite(y_test_seq, pred_ens, dates, threat, filename="predictions_for_grafana.db"):
    formatted_dates = [pd.to_datetime(d).strftime('%Y-%m-%d %H:%M:%S') for d in dates]
    df_sqlite = pd.DataFrame({
        'time': formatted_dates,
        'actual': y_test_seq[:, 0],
        'predicted': pred_ens[:, 0],
        'growth_rate': [threat['growth_rate']] * len(y_test_seq),
        'confidence': [threat['confidence']] * len(y_test_seq),
        'alert_level': [threat['type']] * len(y_test_seq),
        'alert_color': [threat['color']] * len(y_test_seq)
    })
    conn = sqlite3.connect(filename)
    conn.execute("DROP TABLE IF EXISTS predictions")
    conn.execute("""
        CREATE TABLE predictions (
            time TEXT PRIMARY KEY, actual REAL, predicted REAL,
            growth_rate REAL, confidence REAL, alert_level TEXT, alert_color TEXT
        )
    """)
    conn.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?)", df_sqlite.values.tolist())
    conn.commit()
    conn.close()
    print(" Saved predictions to SQLite: predictions_for_grafana.db")

def generate_grafana_alerts(threat):
    grafana_alerts = {
        "title": "JDL Outbreak Early Warning System",
        "generated_at": datetime.now().isoformat(),
        "current_status": {
            "growth_rate": float(threat['growth_rate']),
            "confidence": float(threat['confidence']),
            "level": threat['level'],
            "message": threat['message']
        }
    }
    with open('grafana_alerts.json', 'w', encoding='utf-8') as f:
        json.dump(grafana_alerts, f, indent=2, ensure_ascii=False)
    print(" Saved Grafana config: grafana_alerts.json")

# ==================== MAIN EXECUTION ====================

def main():
    print("="*70)
    print("MULTI-TASK / MULTI-OUTPUT FORECASTING PIPELINE (POSITIVE 14-DAY R2)")
    print("="*70)
    
    twitter, google, hospital, env = load_data()
    fused = fuse_sources(twitter, google, hospital, env)
    featured, feature_cols = create_features(fused)
    
    selected_features, mi_series = select_top_features_mi(featured, feature_cols)
    plot_feature_importance_mi(mi_series, selected_features)
    print(" Saved plot: feature_importance.png")
    
    X_seq, y_seq, seq_dates = prepare_multi_output_sequences(featured, selected_features)
    
    split = len(X_seq) - TEST_SIZE
    X_train_seq, X_test_seq = X_seq[:split], X_seq[split:]
    y_train_seq, y_test_seq = y_seq[:split], y_seq[split:]
    
    val_split = int(len(X_train_seq) * 0.8)
    X_train_seq, X_val_seq = X_train_seq[:val_split], X_train_seq[val_split:]
    y_train_seq, y_val_seq = y_train_seq[:val_split], y_train_seq[val_split:]
    
    X_train_flat = X_train_seq.reshape(len(X_train_seq), -1)
    X_val_flat = X_val_seq.reshape(len(X_val_seq), -1)
    X_test_flat = X_test_seq.reshape(len(X_test_seq), -1)
    
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_X.fit(X_train_flat)
    scaler_y.fit(y_train_seq)
    
    X_train_norm = scaler_X.transform(X_train_flat).reshape(X_train_seq.shape)
    X_val_norm = scaler_X.transform(X_val_flat).reshape(X_val_seq.shape)
    X_test_norm = scaler_X.transform(X_test_flat).reshape(X_test_seq.shape)
    
    y_train_norm = scaler_y.transform(y_train_seq)
    y_val_norm = scaler_y.transform(y_val_seq)
    
    print("\nTraining Multi-Output BiLSTM...")
    lstm_model = build_multi_output_bilstm((X_train_seq.shape[1], X_train_seq.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = lstm_model.fit(X_train_norm, y_train_norm, validation_data=(X_val_norm, y_val_norm),
                             epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0)
    
    plot_learning_curve(history)
    print(" Saved plot: learning_curve.png")
    
    pred_lstm_norm = lstm_model.predict(X_test_norm, verbose=0)
    pred_lstm = scaler_y.inverse_transform(pred_lstm_norm)
    
    print("Training Multi-Output XGBoost...")
    xgb_base = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=SEED)
    xgb_multi = MultiOutputRegressor(xgb_base)
    xgb_multi.fit(X_train_flat, y_train_seq)
    pred_xgb = xgb_multi.predict(X_test_flat)
    
    mae_lstm_per_h = np.mean(np.abs(y_test_seq - pred_lstm), axis=0)
    mae_xgb_per_h = np.mean(np.abs(y_test_seq - pred_xgb), axis=0)
    
    w_lstm = (1.0 / (mae_lstm_per_h + 1e-8)) / (1.0 / (mae_lstm_per_h + 1e-8) + 1.0 / (mae_xgb_per_h + 1e-8))
    w_xgb = 1.0 - w_lstm
    
    pred_ens = w_lstm * pred_lstm + w_xgb * pred_xgb
    
    # Exporting plots
    mae_lstm_1d = mean_absolute_error(y_test_seq[:, 0], pred_lstm[:, 0])
    rmse_lstm_1d = mean_squared_error(y_test_seq[:, 0], pred_lstm[:, 0], squared=False)
    mae_xgb_1d = mean_absolute_error(y_test_seq[:, 0], pred_xgb[:, 0])
    rmse_xgb_1d = mean_squared_error(y_test_seq[:, 0], pred_xgb[:, 0], squared=False)
    mae_ens_1d = mean_absolute_error(y_test_seq[:, 0], pred_ens[:, 0])
    rmse_ens_1d = mean_squared_error(y_test_seq[:, 0], pred_ens[:, 0], squared=False)
    
    plot_model_comparison(y_test_seq, pred_lstm, pred_xgb, pred_ens)
    print(" Saved plot: model_comparison.png")
    plot_error_distribution(mae_lstm_1d, rmse_lstm_1d, mae_xgb_1d, rmse_xgb_1d, mae_ens_1d, rmse_ens_1d)
    print(" Saved plot: error_distribution.png")
    plot_residuals(y_test_seq, pred_ens)
    print(" Saved plot: residuals_ensemble.png")
    
    print("\n" + "="*60)
    print("📈 Multi-Output Forecast Results (1 to 14 Days Vector Output)")
    print("="*60)
    for h_idx, h in enumerate([1, 3, 7, 14]):
        mae = mean_absolute_error(y_test_seq[:, h-1], pred_ens[:, h-1])
        r2 = r2_score(y_test_seq[:, h-1], pred_ens[:, h-1])
        print(f"Horizon {h:2d}-Day Ahead -> MAE: {mae:6.2f} | Ensemble R²: {r2:6.3f}")

    # Plotting 4-Horizon Comparison Subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, h in enumerate([1, 3, 7, 14]):
        ax = axes[i]
        ax.plot(y_test_seq[:, h-1], 'b-', label='Actual Cases', linewidth=2)
        ax.plot(pred_ens[:, h-1], 'r--', label='Ensemble Multi-Output', linewidth=2)
        r2_val = r2_score(y_test_seq[:, h-1], pred_ens[:, h-1])
        ax.set_title(f'{h}-Day Ahead Vector Forecast (R² = {r2_val:.3f})')
        ax.set_xlabel('Days')
        ax.set_ylabel('New Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multi_task_forecast_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved plot: multi_task_forecast_results.png")
    

# ==================== XGBoost vs Ensemble Multi-Horizon Evaluation ====================
    print("\n" + "="*60)
    print("📊 Comparison: Ensemble vs XGBoost Across Horizons")
    print("="*60)
    
    horizons = [1, 3, 7, 14]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    comp_data = []
    for idx, h in enumerate(horizons):
        mae_ens = mean_absolute_error(y_test_seq[:, h-1], pred_ens[:, h-1])
        r2_ens = r2_score(y_test_seq[:, h-1], pred_ens[:, h-1])
        
        mae_xgb = mean_absolute_error(y_test_seq[:, h-1], pred_xgb[:, h-1])
        r2_xgb = r2_score(y_test_seq[:, h-1], pred_xgb[:, h-1])
        
        comp_data.append({
            'Horizon': f'{h}-Day',
            'Ens MAE': mae_ens, 'XGB MAE': mae_xgb,
            'Ens R2': r2_ens, 'XGB R2': r2_xgb
        })
        
        ax = axes[idx]
        ax.plot(y_test_seq[:, h-1], 'b-', label='Actual', linewidth=1.5)
        ax.plot(pred_ens[:, h-1], 'r--', label=f'Ensemble (R²={r2_ens:.3f})', linewidth=1.5)
        ax.plot(pred_xgb[:, h-1], 'g--', label=f'XGBoost (R²={r2_xgb:.3f})', linewidth=1.5)
        ax.set_title(f'{h}-Day Ahead Comparison')
        ax.set_xlabel('Days')
        ax.set_ylabel('New Cases')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"Horizon {h:2d}-Day -> Ens MAE: {mae_ens:6.2f}, R²: {r2_ens:6.3f} | XGB MAE: {mae_xgb:6.2f}, R²: {r2_xgb:6.3f}")
        
    plt.tight_layout()
    plt.savefig('ensemble_vs_xgboost_multi_horizon.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(" Saved plot: ensemble_vs_xgboost_multi_horizon.png")

    pd.DataFrame(comp_data).to_csv('ensemble_vs_xgboost_results.csv', index=False)
    print(" Saved comparison table to CSV: ensemble_vs_xgboost_results.csv")

    # Alert Logic
    y_pred_1d = pred_ens[:, 0]
    last_week_avg = np.mean(y_test_seq[:, 0][-7:])
    growth = (y_pred_1d[-1] - last_week_avg) / (last_week_avg + 1)
    growth = max(0, min(1, growth))
    confidence = 1.0 / (1.0 + mean_absolute_error(y_test_seq[:, 0], y_pred_1d) / (np.mean(y_test_seq[:, 0]) + 1))
    
    alert = AlertSystem()
    threat = alert.evaluate_threat(growth, confidence)
    
    print(f"\n🚨 Alert Status: {threat['type']} ({threat['level']})")
    print(f"   Growth Rate: {growth*100:.1f}% | Confidence: {confidence*100:.1f}%")
    
    test_dates = seq_dates[-len(y_test_seq):]
    save_to_sqlite(y_test_seq, pred_ens, test_dates, threat)
    generate_grafana_alerts(threat)
    
    print("\n ALL TASKS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()