import os
import json
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results.csv")

def log_result(model_name, y_true, y_proba, y_pred, params=None, notes=""):
    """Сохраняет метрики модели в общий results.csv (перезаписывает строку по model_name)."""
    row = {
        "model":     model_name,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "auc":       round(roc_auc_score(y_true, y_proba), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall":    round(recall_score(y_true, y_pred), 4),
        "params":    json.dumps(params or {}, ensure_ascii=False),
        "notes":     notes
    }
    df_new = pd.DataFrame([row])

    if os.path.exists(RESULTS_PATH):
        df_old = pd.read_csv(RESULTS_PATH)
        df_old = df_old[df_old["model"] != model_name]
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(RESULTS_PATH, index=False)
    print(f"✓ {model_name}: AUC={row['auc']}, acc={row['accuracy']} → {RESULTS_PATH}")
    return row


def load_results(sort_by="auc"):
    """Читает results.csv и возвращает отсортированную таблицу."""
    if not os.path.exists(RESULTS_PATH):
        return pd.DataFrame()
    return pd.read_csv(RESULTS_PATH).sort_values(sort_by, ascending=False)