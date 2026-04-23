"""
阶段 2A：Track A — 院内死亡率预测
输入：features_track_a.csv
输出：results_track_a.pkl（供 2-DB 仪表板使用）
运行：python stage2a_model.py
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline
import xgboost as xgb

# ════════════════════════════════════════════════════════
# 加载数据
# ════════════════════════════════════════════════════════

def load_features():
    df = pd.read_csv("features_track_a.csv")

    TARGET = "hospital_expire_flag"
    DROP   = ["hadm_id", "subject_id", "admittime", "dischtime", TARGET]
    feat_cols = [c for c in df.columns if c not in DROP]

    X = df[feat_cols].copy()
    y = df[TARGET].copy()

    # 填充残余缺失（保险起见）
    X = X.fillna(X.median(numeric_only=True))

    print(f"  样本数：{len(X)}  特征数：{len(feat_cols)}")
    print(f"  死亡率：{y.mean()*100:.1f}%  (死亡={y.sum()}, 存活={len(y)-y.sum()})")
    print(f"  特征列：{feat_cols}")
    return X, y, feat_cols


# ════════════════════════════════════════════════════════
# 模型训练
# ════════════════════════════════════════════════════════

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

    models = {
        "逻辑回归": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                C=0.1,
            )),
        ]),
        "随机森林": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric="auc",
            random_state=42,
            verbosity=0,
        ),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n  ── {name} ──")
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auroc = roc_auc_score(y_test, y_prob)
        ap    = average_precision_score(y_test, y_prob)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        cm    = confusion_matrix(y_test, y_pred)
        cr    = classification_report(y_test, y_pred,
                                      target_names=["存活", "死亡"],
                                      output_dict=True)

        # 5-fold CV AUROC
        cv_scores = cross_val_score(model, X, y, cv=cv,
                                    scoring="roc_auc", n_jobs=-1)

        print(f"    AUROC:      {auroc:.3f}")
        print(f"    AP:         {ap:.3f}")
        print(f"    CV AUROC:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"    死亡 Recall: {cr['死亡']['recall']:.3f}")
        print(f"    死亡 F1:     {cr['死亡']['f1-score']:.3f}")
        print(f"    混淆矩阵:\n{cm}")

        # 特征重要性
        feat_imp = _get_feature_importance(model, name, X.columns.tolist())

        results[name] = {
            "model":       model,
            "auroc":       auroc,
            "ap":          ap,
            "cv_auroc":    cv_scores,
            "fpr":         fpr,
            "tpr":         tpr,
            "cm":          cm,
            "report":      cr,
            "feat_imp":    feat_imp,
            "y_test":      y_test.values,
            "y_prob":      y_prob,
            "y_pred":      y_pred,
            "feat_cols":   X.columns.tolist(),
        }

    return results, X_test, y_test


def _get_feature_importance(model, name, feat_cols):
    if name == "逻辑回归":
        coef = model.named_steps["clf"].coef_[0]
        imp  = pd.Series(np.abs(coef), index=feat_cols)
    elif name == "随机森林":
        imp = pd.Series(model.feature_importances_, index=feat_cols)
    elif name == "XGBoost":
        imp = pd.Series(model.feature_importances_, index=feat_cols)
    else:
        return None
    return imp.sort_values(ascending=False)


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("阶段 2A：Track A 死亡率预测建模")
    print("=" * 55)

    print("\n[1] 加载特征矩阵 ...")
    X, y, feat_cols = load_features()

    print("\n[2] 训练模型 ...")
    results, X_test, y_test = train_models(X, y)

    print("\n[3] 保存结果 ...")
    with open("results_track_a.pkl", "wb") as f:
        pickle.dump(results, f)
    print("  ✓ 已保存 results_track_a.pkl")

    best = max(results.items(), key=lambda x: x[1]["auroc"])
    print(f"\n✅ 2A 完成！最佳模型：{best[0]}  AUROC={best[1]['auroc']:.3f}")
    return results


if __name__ == "__main__":
    main()