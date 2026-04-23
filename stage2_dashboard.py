
"""
阶段 2-DB：Level 2 建模结果 HTML 仪表板
输入：results_track_a.pkl, results_track_b.pkl
输出：modeling_dashboard.html
运行：python stage2_dashboard.py
"""

import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


# ════════════════════════════════════════════════════════
# Track A 图表
# ════════════════════════════════════════════════════════

def fig_roc_curves(results_a):
    """ROC 曲线对比（三个模型）"""
    COLORS = {"逻辑回归": "#5B8DEF", "随机森林": "#4ECDC4", "XGBoost": "#FF6B6B"}
    fig = go.Figure()

    for name, r in results_a.items():
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"],
            mode="lines",
            name=f"{name} (AUROC={r['auroc']:.3f})",
            line=dict(color=COLORS.get(name, "#888"), width=2),
            hovertemplate=f"{name}<br>FPR=%{{x:.3f}}<br>TPR=%{{y:.3f}}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random",
        line=dict(color="#555", width=1, dash="dash"),
        showlegend=True,
    ))
    fig.update_layout(
        title=dict(text="ROC 曲线对比", font=dict(size=15, color="#e8eaf0")),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf0"),
        margin=dict(t=50, b=40, l=20, r=20),
        height=360,
    )
    return fig


def fig_confusion_matrix(results_a):
    """最佳模型的混淆矩阵（XGBoost）"""
    # 优先取 XGBoost，否则取第一个
    name = "XGBoost" if "XGBoost" in results_a else list(results_a.keys())[0]
    r  = results_a[name]
    cm = r["cm"]

    labels = ["存活", "死亡"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=["预测：存活", "预测：死亡"],
        y=["实际：存活", "实际：死亡"],
        colorscale=[[0, "#1a1d2e"], [1, "#FF6B6B"]],
        showscale=False,
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=22, color="#fff"),
        hovertemplate="实际：%{y}<br>预测：%{x}<br>数量：%{z}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"混淆矩阵（{name}）", font=dict(size=15, color="#e8eaf0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf0"),
        margin=dict(t=50, b=40, l=20, r=20),
        height=300,
    )
    return fig


def fig_feature_importance_a(results_a):
    """Track A 特征重要性（Top 15）"""
    name = "XGBoost" if "XGBoost" in results_a else list(results_a.keys())[0]
    imp  = results_a[name]["feat_imp"].head(15).sort_values()

    # 特征名映射（更可读）
    NAME_MAP = {
        "age": "年龄", "age_imputed": "年龄缺失标记", "gender_M": "性别(男)",
        "adm_EMERGENCY": "急诊入院", "adm_ELECTIVE": "择期入院", "adm_URGENT": "紧急入院",
        "charlson_index": "Charlson指数",
        "creatinine_min": "肌酐_min", "creatinine_max": "肌酐_max", "creatinine_mean": "肌酐_mean",
        "bun_min": "BUN_min", "bun_max": "BUN_max", "bun_mean": "BUN_mean",
        "wbc_min": "WBC_min", "wbc_max": "WBC_max", "wbc_mean": "WBC_mean",
        "platelets_min": "血小板_min", "platelets_max": "血小板_max", "platelets_mean": "血小板_mean",
        "hemoglobin_min": "血红蛋白_min", "hemoglobin_max": "血红蛋白_max", "hemoglobin_mean": "血红蛋白_mean",
        "bilirubin_min": "胆红素_min", "bilirubin_max": "胆红素_max", "bilirubin_mean": "胆红素_mean",
        "lactate_min": "乳酸_min", "lactate_max": "乳酸_max", "lactate_mean": "乳酸_mean",
    }
    labels = [NAME_MAP.get(f, f) for f in imp.index]

    fig = go.Figure(go.Bar(
        x=imp.values,
        y=labels,
        orientation="h",
        marker=dict(
            color=imp.values,
            colorscale=[[0, "#2a3a6a"], [1, "#5B8DEF"]],
            showscale=False,
        ),
        hovertemplate="%{y}：%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"特征重要性 Top 15（{name}）", font=dict(size=15, color="#e8eaf0")),
        xaxis_title="重要性分数",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf0"),
        yaxis=dict(automargin=True, tickfont=dict(size=11)),
        margin=dict(t=50, b=40, l=20, r=20),
        height=400,
    )
    return fig


def make_metric_cards_a(results_a):
    """Track A 指标卡片 HTML"""
    cards = []
    for name, r in results_a.items():
        cr = r["report"]
        recall_death = cr.get("死亡", {}).get("recall", 0)
        f1_death     = cr.get("死亡", {}).get("f1-score", 0)
        cv_mean      = r.get("cv_auroc", np.array([r["auroc"]])).mean()
        cards.append(f"""
        <div class="metric-group">
          <div class="model-name">{name}</div>
          <div class="metrics-row">
            <div class="metric-card">
              <div class="metric-val">{r['auroc']:.3f}</div>
              <div class="metric-label">AUROC</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{cv_mean:.3f}</div>
              <div class="metric-label">CV AUROC</div>
            </div>
            <div class="metric-card highlight">
              <div class="metric-val">{recall_death:.3f}</div>
              <div class="metric-label">死亡 Recall</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{f1_death:.3f}</div>
              <div class="metric-label">死亡 F1</div>
            </div>
          </div>
        </div>""")
    return "\n".join(cards)


# ════════════════════════════════════════════════════════
# Track B 图表
# ════════════════════════════════════════════════════════

def fig_pred_vs_actual(results_b):
    """预测值 vs 实际值散点图"""
    COLORS = {"Baseline": "#888", "XGBoost": "#FF6B6B", "LightGBM": "#4ECDC4"}
    fig = go.Figure()

    y_max = 0
    for name, r in results_b.items():
        y_true = r["y_test_orig"]
        y_pred = r["y_pred"]
        y_max  = max(y_max, y_true.max(), y_pred.max())

        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode="markers",
            name=f"{name} (MAE={r['mae']:.2f}d)",
            marker=dict(color=COLORS.get(name, "#888"), size=7, opacity=0.65),
            hovertemplate=f"{name}<br>实际：%{{x:.2f}}天<br>预测：%{{y:.2f}}天<extra></extra>",
        ))

    # 对角线（完美预测线）
    fig.add_trace(go.Scatter(
        x=[0, y_max], y=[0, y_max],
        mode="lines",
        name="完美预测",
        line=dict(color="#FFD93D", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=dict(text="预测 vs 实际 LOS", font=dict(size=15, color="#e8eaf0")),
        xaxis_title="实际 LOS（天）",
        yaxis_title="预测 LOS（天）",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf0"),
        legend=dict(x=0.02, y=0.95, bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=40, l=20, r=20),
        height=360,
    )
    return fig


def fig_error_distribution(results_b):
    """误差分布直方图"""
    COLORS = {"Baseline": "#888", "XGBoost": "#FF6B6B", "LightGBM": "#4ECDC4"}
    fig = go.Figure()

    for name, r in results_b.items():
        errors = r["y_pred"] - r["y_test_orig"]
        fig.add_trace(go.Histogram(
            x=errors,
            name=name,
            opacity=0.65,
            nbinsx=30,
            marker_color=COLORS.get(name, "#888"),
            hovertemplate=f"{name}<br>误差：%{{x:.2f}}天<br>频次：%{{y}}<extra></extra>",
        ))

    fig.add_vline(x=0, line_dash="dash", line_color="#FFD93D", line_width=1.5,
                  annotation_text="零误差", annotation_font_color="#FFD93D")
    fig.update_layout(
        title=dict(text="预测误差分布（预测 − 实际）", font=dict(size=15, color="#e8eaf0")),
        xaxis_title="误差（天）",
        yaxis_title="频次",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf0"),
        margin=dict(t=50, b=40, l=20, r=20),
        height=320,
    )
    return fig


def fig_feature_importance_b(results_b):
    """Track B 特征重要性（XGBoost 或 LightGBM Top 15）"""
    name = "LightGBM" if "LightGBM" in results_b else "XGBoost"
    if results_b[name]["feat_imp"] is None:
        return go.Figure()

    imp = results_b[name]["feat_imp"].head(15).sort_values()

    NAME_MAP = {
        "age": "年龄", "age_imputed": "年龄缺失标记", "gender_M": "性别(男)",
        "adm_EMERGENCY": "急诊入院", "adm_ELECTIVE": "择期入院", "adm_URGENT": "紧急入院",
        "charlson_index": "Charlson指数", "diag_count": "诊断数量",
        "hr_min": "心率_min", "hr_max": "心率_max", "hr_mean": "心率_mean",
        "sbp_min": "收缩压_min", "sbp_max": "收缩压_max", "sbp_mean": "收缩压_mean",
        "dbp_min": "舒张压_min", "dbp_max": "舒张压_max", "dbp_mean": "舒张压_mean",
        "spo2_min": "血氧_min", "spo2_max": "血氧_max", "spo2_mean": "血氧_mean",
        "temp_min": "体温_min", "temp_max": "体温_max", "temp_mean": "体温_mean",
        "rr_min": "呼吸率_min", "rr_max": "呼吸率_max", "rr_mean": "呼吸率_mean",
    }
    labels = [NAME_MAP.get(f, f) for f in imp.index]

    fig = go.Figure(go.Bar(
        x=imp.values,
        y=labels,
        orientation="h",
        marker=dict(
            color=imp.values,
            colorscale=[[0, "#1a3a2a"], [1, "#4ECDC4"]],
            showscale=False,
        ),
        hovertemplate="%{y}：%{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"特征重要性 Top 15（{name}）", font=dict(size=15, color="#e8eaf0")),
        xaxis_title="重要性分数",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,46,0.6)",
        font=dict(color="#e8eaf0"),
        yaxis=dict(automargin=True, tickfont=dict(size=11)),
        margin=dict(t=50, b=40, l=20, r=20),
        height=400,
    )
    return fig


def make_metric_cards_b(results_b):
    """Track B 指标卡片 HTML"""
    cards = []
    for name, r in results_b.items():
        cv_str = ""
        if "cv_mae" in r:
            cv_str = f"""
            <div class="metric-card">
              <div class="metric-val">{r['cv_mae'].mean():.3f}</div>
              <div class="metric-label">CV MAE</div>
            </div>"""
        r2_str = f"{r['r2']:.3f}" if r["r2"] > -999 else "—"
        cards.append(f"""
        <div class="metric-group">
          <div class="model-name">{name}</div>
          <div class="metrics-row">
            <div class="metric-card highlight">
              <div class="metric-val">{r['mae']:.3f}</div>
              <div class="metric-label">MAE（天）</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r['rmse']:.3f}</div>
              <div class="metric-label">RMSE（天）</div>
            </div>
            <div class="metric-card">
              <div class="metric-val">{r2_str}</div>
              <div class="metric-label">R²</div>
            </div>
            {cv_str}
          </div>
        </div>""")
    return "\n".join(cards)


# ════════════════════════════════════════════════════════
# 组装 HTML
# ════════════════════════════════════════════════════════

def build_dashboard(results_a, results_b, output_path="modeling_dashboard.html"):
    def to_div(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    # Track A
    roc_div  = to_div(fig_roc_curves(results_a))
    cm_div   = to_div(fig_confusion_matrix(results_a))
    impa_div = to_div(fig_feature_importance_a(results_a))
    cards_a  = make_metric_cards_a(results_a)

    # Track B
    pva_div  = to_div(fig_pred_vs_actual(results_b))
    err_div  = to_div(fig_error_distribution(results_b))
    impb_div = to_div(fig_feature_importance_b(results_b))
    cards_b  = make_metric_cards_b(results_b)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <title>MIMIC-III Modeling Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
      background: #0f1117;
      color: #e8eaf0;
      min-height: 100vh;
    }}
    header {{
      padding: 24px 40px 16px;
      border-bottom: 1px solid #2a2d3a;
      background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
    }}
    header h1 {{ font-size: 20px; font-weight: 700; color: #fff; }}
    header p  {{ font-size: 12px; color: #888; margin-top: 4px; }}

    .track-section {{
      padding: 28px 40px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    .track-title {{
      font-size: 16px;
      font-weight: 700;
      color: #fff;
      padding: 10px 16px;
      border-radius: 8px;
      margin-bottom: 20px;
    }}
    .track-a .track-title {{ background: linear-gradient(90deg, #1e3a6e, #1a1d2e); border-left: 4px solid #5B8DEF; }}
    .track-b .track-title {{ background: linear-gradient(90deg, #1a3a2a, #1a1d2e); border-left: 4px solid #4ECDC4; }}

    /* 指标卡片 */
    .metric-group {{ margin-bottom: 20px; }}
    .model-name {{
      font-size: 13px; font-weight: 600; color: #aaa;
      margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .metrics-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .metric-card {{
      background: #1a1d2e;
      border: 1px solid #2a2d3a;
      border-radius: 10px;
      padding: 14px 20px;
      min-width: 110px;
      text-align: center;
    }}
    .metric-card.highlight {{ border-color: #FF6B6B; }}
    .metric-val {{ font-size: 22px; font-weight: 700; color: #fff; }}
    .metric-label {{ font-size: 11px; color: #888; margin-top: 4px; }}

    /* 图表网格 */
    .chart-grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .chart-grid-3 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 20px;
      margin-bottom: 20px;
    }}
    .chart-card {{
      background: #1a1d2e;
      border: 1px solid #2a2d3a;
      border-radius: 12px;
      padding: 16px;
      overflow: hidden;
    }}
    .chart-card:hover {{ border-color: #5B8DEF; }}
    .track-b .chart-card:hover {{ border-color: #4ECDC4; }}

    .divider {{
      border: none;
      border-top: 1px solid #2a2d3a;
      margin: 8px 40px;
    }}
  </style>
</head>
<body>

<header>
  <h1>MIMIC-III — Level 2 Modeling Dashboard</h1>
  <p>Track A: 院内死亡率预测（分类）&nbsp;·&nbsp;Track B: ICU 住院时长预测（回归）</p>
</header>

<!-- ════ Track A ════ -->
<div class="track-section track-a">
  <div class="track-title">Track A — 院内死亡率预测（Binary Classification）</div>

  {cards_a}

  <div class="chart-grid-2">
    <div class="chart-card">{roc_div}</div>
    <div class="chart-card">{cm_div}</div>
  </div>
  <div class="chart-card">{impa_div}</div>
</div>

<hr class="divider">

<!-- ════ Track B ════ -->
<div class="track-section track-b">
  <div class="track-title">Track B — ICU 住院时长预测（Regression）</div>

  {cards_b}

  <div class="chart-grid-2">
    <div class="chart-card">{pva_div}</div>
    <div class="chart-card">{err_div}</div>
  </div>
  <div class="chart-card">{impb_div}</div>
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ 仪表板已保存：{output_path}")


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("阶段 2-DB：建模仪表板生成")
    print("=" * 55)

    print("\n[1] 加载 Track A 结果 ...")
    with open("results_track_a.pkl", "rb") as f:
        results_a = pickle.load(f)
    print(f"  模型：{list(results_a.keys())}")

    print("\n[2] 加载 Track B 结果 ...")
    with open("results_track_b.pkl", "rb") as f:
        results_b = pickle.load(f)
    print(f"  模型：{list(results_b.keys())}")

    print("\n[3] 生成仪表板 ...")
    build_dashboard(results_a, results_b)


if __name__ == "__main__":
    main()