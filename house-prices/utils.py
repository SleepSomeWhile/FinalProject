import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results.csv")

# y_true_log, y_pred_log : array, optional Если переданы, считается RMSLE (метрика Kaggle). 
# Если нет — RMSLE аппроксимируется как RMSE на log1p.
def log_result(
    model_name,
    y_true,
    y_pred,
    y_true_log=None,
    y_pred_log=None,
    params=None,
    notes="",
):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()


    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))


    if y_true_log is not None and y_pred_log is not None:
        yt = np.asarray(y_true_log).ravel()
        yp = np.asarray(y_pred_log).ravel()
    else:
        yt = np.log1p(np.clip(y_true, 0, None))
        yp = np.log1p(np.clip(y_pred, 0, None))
    rmsle = float(np.sqrt(mean_squared_error(yt, yp)))


    mape = float(mean_absolute_percentage_error(y_true, y_pred)) * 100

    row = {
        "model":     model_name,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "rmsle":     round(rmsle, 4),      
        "rmse":      round(rmse, 2),       
        "mae":       round(mae, 2),        
        "r2":        round(r2, 4),
        "mape_pct":  round(mape, 2),
        "params":    json.dumps(params or {}, ensure_ascii=False, default=str),
        "notes":     notes,
    }
    df_new = pd.DataFrame([row])

    if os.path.exists(RESULTS_PATH):
        df_old = pd.read_csv(RESULTS_PATH)
        df_old = df_old[df_old["model"] != model_name]      # перезапись по имени
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(RESULTS_PATH, index=False)
    print(f"✓ {model_name}: RMSLE={row['rmsle']}, R²={row['r2']}, RMSE=${row['rmse']:.0f}")
    return row


def load_results(sort_by="rmsle", ascending=True):
    """Читает results.csv и возвращает отсортированную таблицу (по RMSLE — меньше лучше)."""
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH).sort_values(sort_by, ascending=ascending)