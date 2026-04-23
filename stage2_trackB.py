"""
阶段 2B：Track B — ICU 住院时长预测
输入：features_track_b.csv
输出：results_track_b.pkl（供 2-DB 仪表板使用）
运行：python stage2b_model.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
import xgboost as xgb
import lightgbm as lgb


# ════════════════════════════════════════════════════════
# 加载数据
# ════════════════════════════════════════════════════════

def load_features():
    df = pd.read_csv("features_track_b.csv")

    TARGET = "los"
    DROP   = ["icustay_id", "hadm_id", "subject_id", "admittime", TARGET]
    feat_cols = [c for c in df.columns if c not in DROP]

    # 过滤异常 LOS（负数或 0）
    df = df[df[TARGET] > 0].copy()

    X = df[feat_cols].copy()
    y = df[TARGET].copy()

    X = X.fillna(X.median(numeric_only=True))

    print(f"  样本数：{len(X)}  特征数：{len(feat_cols)}")
    print(f"  LOS 分布：中位数={y.median():.2f}天  均值={y.mean():.2f}天  最大={y.max():.2f}天")
    print(f"  特征列：{feat_cols}")
    return X, y, feat_cols


# ════════════════════════════════════════════════════════
# 评估函数
# ════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"    MAE:  {mae:.3f} 天")
    print(f"    RMSE: {rmse:.3f} 天")
    print(f"    R²:   {r2:.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "r2": r2}


# ════════════════════════════════════════════════════════
# 模型训练
# ════════════════════════════════════════════════════════

def train_models(X, y):
    # log1p 变换目标变量（右偏分布）
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    y_test_orig = np.expm1(y_test)  # 反变换，用于最终评估

    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

    results = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # ── Baseline：均值预测 ──────────────────────────────
    print("\n  ── Baseline（均值预测）──")
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    y_pred_dummy = np.expm1(dummy.predict(X_test))
    metrics_dummy = evaluate(y_test_orig, y_pred_dummy, "Baseline")
    results["Baseline"] = {
        **metrics_dummy,
        "y_pred": y_pred_dummy,
        "feat_imp": None,
    }

    # ── XGBoost ────────────────────────────────────────
    print("\n  ── XGBoost ──")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    y_pred_xgb = np.expm1(xgb_model.predict(X_test))
    metrics_xgb = evaluate(y_test_orig, y_pred_xgb, "XGBoost")

    cv_scores_xgb = -cross_val_score(
        xgb_model, X, y_log, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    print(f"    CV MAE: {cv_scores_xgb.mean():.3f} ± {cv_scores_xgb.std():.3f}")

    feat_imp_xgb = pd.Series(
        xgb_model.feature_importances_,
        index=X.columns.tolist()
    ).sort_values(ascending=False)

    results["XGBoost"] = {
        **metrics_xgb,
        "cv_mae":  cv_scores_xgb,
        "y_pred":  y_pred_xgb,
        "feat_imp": feat_imp_xgb,
    }

    # ── LightGBM ───────────────────────────────────────
    print("\n  ── LightGBM ──")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    y_pred_lgb = np.expm1(lgb_model.predict(X_test))
    metrics_lgb = evaluate(y_test_orig, y_pred_lgb, "LightGBM")

    cv_scores_lgb = -cross_val_score(
        lgb_model, X, y_log, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    print(f"    CV MAE: {cv_scores_lgb.mean():.3f} ± {cv_scores_lgb.std():.3f}")

    feat_imp_lgb = pd.Series(
        lgb_model.feature_importances_,
        index=X.columns.tolist()
    ).sort_values(ascending=False)

    results["LightGBM"] = {
        **metrics_lgb,
        "cv_mae":  cv_scores_lgb,
        "y_pred":  y_pred_lgb,
        "feat_imp": feat_imp_lgb,
    }

    # 共用数据存入每个结果
    for r in results.values():
        r["y_test_orig"] = y_test_orig.values
        r["feat_cols"]   = X.columns.tolist()

    return results, X_test, y_test_orig


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("阶段 2B：Track B ICU LOS 预测建模")
    print("=" * 55)

    print("\n[1] 加载特征矩阵 ...")
    X, y, feat_cols = load_features()

    print("\n[2] 训练模型 ...")
    results, X_test, y_test_orig = train_models(X, y)

    print("\n[3] 汇总对比 ...")
    print(f"\n  {'模型':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("  " + "-" * 38)
    for name, r in results.items():
        print(f"  {name:<12} {r['mae']:>8.3f} {r['rmse']:>8.3f} {r['r2']:>8.3f}")

    print("\n[4] 保存结果 ...")
    with open("results_track_b.pkl", "wb") as f:
        pickle.dump(results, f)
    print("  ✓ 已保存 results_track_b.pkl")

    best = min(
        [(k, v) for k, v in results.items() if k != "Baseline"],
        key=lambda x: x[1]["mae"]
    )
    print(f"\n✅ 2B 完成！最佳模型：{best[0]}  MAE={best[1]['mae']:.3f}天")
    return results


if __name__ == "__main__":
    main()