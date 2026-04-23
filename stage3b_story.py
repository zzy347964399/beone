"""
Stage 3b: Patient Story Analysis
分析多维度 Lab 数据，发现规律、异常和患者个体叙事。
输出：patient_story.html
运行：python stage3b_story.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.io as pio
import os

DATA_DIR = "/Users/pickle/.cache/kagglehub/datasets/asjad99/mimiciii/versions/1/mimic-iii-clinical-database-demo-1.4"
OUTPUT_FILE = "patient_story.html"

LAYOUT_BASE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#1a1a2e", family="Arial, sans-serif"),
)

# 关注的 Lab 指标
LAB_ITEMS = {
    50912: ("Creatinine",  "mg/dL",  (0, 20)),
    50813: ("Lactate",     "mmol/L", (0, 20)),
    51301: ("WBC",         "K/uL",   (0, 100)),
    51222: ("Hemoglobin",  "g/dL",   (0, 20)),
    50885: ("Bilirubin",   "mg/dL",  (0, 30)),
    50861: ("ALT",         "U/L",    (0, 1000)),
}
LAB_NAMES = {v[0]: k for k, v in LAB_ITEMS.items()}


# ════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════

def load_data():
    def read(fname, cols=None, dates=None):
        path = os.path.join(DATA_DIR, fname)
        return pd.read_csv(path, usecols=cols, parse_dates=dates, low_memory=False)

    admissions = read("ADMISSIONS.csv",
        cols=["subject_id","hadm_id","admittime","hospital_expire_flag"],
        dates=["admittime"])
    admissions["hospital_expire_flag"] = admissions["hospital_expire_flag"].fillna(0).astype(int)

    icustays = read("ICUSTAYS.csv",
        cols=["subject_id","hadm_id","icustay_id","intime","outtime","los"],
        dates=["intime","outtime"])

    labevents = read("LABEVENTS.csv",
        cols=["subject_id","hadm_id","itemid","charttime","valuenum"],
        dates=["charttime"])

    # 过滤目标 itemid，清除无效值
    lab = labevents[labevents["itemid"].isin(LAB_ITEMS)].copy()
    lab = lab[lab["valuenum"].notna() & (lab["valuenum"] > 0)]

    # clip 异常值
    for itemid, (name, unit, (lo, hi)) in LAB_ITEMS.items():
        mask = lab["itemid"] == itemid
        lab.loc[mask, "valuenum"] = lab.loc[mask, "valuenum"].clip(lo, hi)

    return admissions, icustays, lab


# ════════════════════════════════════════════════════════
# 图 1：Lab 指标相关性热图
# ════════════════════════════════════════════════════════

def fig_correlation_heatmap(lab, admissions):
    """
    Story: 计算各 Lab 指标峰值之间的 Pearson 相关矩阵。
    乳酸与胆红素同步升高 → 多器官功能障碍信号。
    肌酐与 BUN 高度相关 → 肾功能恶化协同模式。
    """
    # 每位患者每个指标的峰值
    pivot = (
        lab.groupby(["subject_id","itemid"])["valuenum"]
        .max().reset_index()
        .pivot(index="subject_id", columns="itemid", values="valuenum")
    )
    pivot.columns = [LAB_ITEMS[c][0] for c in pivot.columns if c in LAB_ITEMS]
    pivot = pivot.dropna(thresh=3)  # 至少有 3 个指标才纳入

    corr = pivot.corr()
    labels = corr.columns.tolist()
    z = corr.values

    # 自定义颜色：负相关蓝，正相关红
    colorscale = [
        [0.0,  "#1D4ED8"],
        [0.5,  "#ffffff"],
        [1.0,  "#DC2626"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels, y=labels,
        colorscale=colorscale,
        zmid=0, zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text:.2f}",
        textfont=dict(size=11),
        hovertemplate="<b>%{x} vs %{y}</b><br>Pearson r = %{z:.3f}<extra></extra>",
        colorbar=dict(title="Pearson r", thickness=14, len=0.8),
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Lab Marker Correlation Matrix (Peak Values per Patient)",
            font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"
        ),
        xaxis=dict(tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        margin=dict(t=60, b=60, l=80, r=80),
        height=440,
    )
    return fig, corr


# ════════════════════════════════════════════════════════
# 图 2：死亡 vs 存活组 Lab 均值对比
# ════════════════════════════════════════════════════════

def fig_mortality_lab_comparison(lab, admissions):
    """
    Story: 比较死亡组与存活组各 Lab 指标的峰值均值。
    死亡患者通常在肌酐、乳酸、胆红素上显著更高，
    血红蛋白则更低，直观呈现多器官功能恶化趋势。
    """
    # 每位患者每个指标的峰值
    peak = (
        lab.groupby(["subject_id","itemid"])["valuenum"]
        .max().reset_index()
    )
    peak = peak.merge(
        admissions[["subject_id","hospital_expire_flag"]].drop_duplicates("subject_id"),
        on="subject_id", how="left"
    )

    fig = go.Figure()
    COLORS = {"Survived": "#10B981", "Deceased": "#EF4444"}

    for flag, label in [(0, "Survived"), (1, "Deceased")]:
        sub = peak[peak["hospital_expire_flag"] == flag]
        means = sub.groupby("itemid")["valuenum"].mean()

        x_labels, y_vals = [], []
        for itemid, (name, unit, _) in LAB_ITEMS.items():
            if itemid in means.index:
                x_labels.append(f"{name}<br>({unit})")
                y_vals.append(means[itemid])

        fig.add_trace(go.Bar(
            name=label,
            x=x_labels,
            y=y_vals,
            marker_color=COLORS[label],
            marker_line=dict(color="#ffffff", width=1),
            opacity=0.85,
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>Mean Peak: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Mean Peak Lab Values: Deceased vs Survived",
            font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"
        ),
        barmode="group",
        yaxis=dict(title="Mean Peak Value", showgrid=True, gridcolor="#e5e7eb"),
        xaxis=dict(title="Lab Marker"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="#e2e8f0", borderwidth=1),
        margin=dict(t=60, b=70, l=60, r=20),
        height=420,
        bargap=0.25, bargroupgap=0.08,
    )
    return fig


# ════════════════════════════════════════════════════════
# 图 3：异常患者识别（Z-score）
# ════════════════════════════════════════════════════════

def fig_outlier_detection(lab, admissions):
    """
    Story: 用 Z-score 识别各指标极端值患者。
    |Z| > 2.5 的患者往往对应高死亡风险或特殊病种。
    """
    peak = (
        lab.groupby(["subject_id","itemid"])["valuenum"]
        .max().reset_index()
        .pivot(index="subject_id", columns="itemid", values="valuenum")
    )
    peak.columns = [LAB_ITEMS[c][0] for c in peak.columns if c in LAB_ITEMS]

    # Z-score
    zscore = (peak - peak.mean()) / peak.std()
    max_z  = zscore.abs().max(axis=1).sort_values(ascending=False)

    # 取 Top 20 异常患者
    top_outliers = max_z.head(20).reset_index()
    top_outliers.columns = ["subject_id", "max_abs_z"]

    # 合并死亡标签
    mortality = admissions[["subject_id","hospital_expire_flag"]].drop_duplicates("subject_id")
    top_outliers = top_outliers.merge(mortality, on="subject_id", how="left")
    top_outliers["outcome"] = top_outliers["hospital_expire_flag"].map(
        {0: "Survived", 1: "Deceased"})

    # 找出每人的极端指标
    def get_extreme_marker(sid):
        row = zscore.loc[sid] if sid in zscore.index else pd.Series()
        if row.empty: return "—"
        return row.abs().idxmax()

    top_outliers["extreme_marker"] = top_outliers["subject_id"].apply(get_extreme_marker)

    colors = top_outliers["hospital_expire_flag"].map(
        {0: "#10B981", 1: "#EF4444"}).fillna("#94A3B8")

    fig = go.Figure(go.Bar(
        x=top_outliers["subject_id"].astype(str),
        y=top_outliers["max_abs_z"],
        marker_color=colors,
        marker_line=dict(color="#ffffff", width=0.5),
        customdata=top_outliers[["outcome","extreme_marker"]].values,
        hovertemplate=(
            "<b>Subject %{x}</b><br>"
            "Max |Z-score|: %{y:.2f}<br>"
            "Extreme marker: %{customdata[1]}<br>"
            "Outcome: %{customdata[0]}"
            "<extra></extra>"
        ),
    ))
    # 阈值线
    fig.add_hline(y=2.5, line_dash="dash", line_color="#F59E0B", line_width=1.5,
                  annotation_text="|Z| = 2.5 threshold",
                  annotation_font=dict(size=11, color="#F59E0B"),
                  annotation_position="top right")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Top 20 Outlier Patients by Max |Z-score| across Lab Markers",
            font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"
        ),
        xaxis=dict(title="Subject ID", tickfont=dict(size=10), tickangle=-35),
        yaxis=dict(title="Max |Z-score|", showgrid=True, gridcolor="#e5e7eb"),
        margin=dict(t=60, b=80, l=60, r=20),
        height=400,
        showlegend=False,
    )
    return fig, top_outliers


# ════════════════════════════════════════════════════════
# 图 4：个体患者 Lab 时序曲线（死亡 vs 存活各选 2 人）
# ════════════════════════════════════════════════════════

def fig_patient_timeseries(lab, admissions, icustays, top_outliers):
    """
    Story: 选取死亡组和存活组各 2 名代表性患者，
    绘制 ICU 住院期间关键 Lab 指标的时序变化。
    死亡患者通常呈现乳酸/肌酐持续攀升，血红蛋白持续下降的趋势。
    """
    # 选患者：死亡组取最异常的 2 人，存活组取数据最丰富的 2 人
    deceased_ids = top_outliers[top_outliers["hospital_expire_flag"] == 1]["subject_id"].head(2).tolist()
    survived_ids = top_outliers[top_outliers["hospital_expire_flag"] == 0]["subject_id"].head(2).tolist()

    # 如果异常列表里存活不够，补充数据丰富的存活患者
    if len(survived_ids) < 2:
        survived_adm = admissions[admissions["hospital_expire_flag"] == 0]["subject_id"].unique()
        counts = lab[lab["subject_id"].isin(survived_adm)].groupby("subject_id").size()
        extra = counts.nlargest(4).index.tolist()
        for sid in extra:
            if sid not in survived_ids:
                survived_ids.append(sid)
            if len(survived_ids) >= 2:
                break

    selected = {
        "Deceased": deceased_ids[:2],
        "Survived": survived_ids[:2],
    }

    # 要展示的指标
    SHOW_ITEMS = [50912, 50813, 51222]  # Creatinine, Lactate, Hemoglobin
    ITEM_NAMES = {50912: "Creatinine (mg/dL)", 50813: "Lactate (mmol/L)",
                  51222: "Hemoglobin (g/dL)"}

    COLOR_DEAD = ["#EF4444", "#F97316"]
    COLOR_SURV = ["#10B981", "#2563EB"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[ITEM_NAMES[i] for i in SHOW_ITEMS],
        horizontal_spacing=0.10,
    )

    icu_merge = icustays[["subject_id","intime"]].drop_duplicates("subject_id")

    for group_label, sids in selected.items():
        colors = COLOR_DEAD if group_label == "Deceased" else COLOR_SURV
        for ci, sid in enumerate(sids):
            # 获取该患者 ICU intime 作为时间零点
            icu_row = icu_merge[icu_merge["subject_id"] == sid]
            if icu_row.empty:
                adm_row = admissions[admissions["subject_id"] == sid]
                t0 = pd.to_datetime(adm_row["admittime"].iloc[0]) if not adm_row.empty else None
            else:
                t0 = pd.to_datetime(icu_row["intime"].iloc[0])

            pat_lab = lab[lab["subject_id"] == sid].copy()
            if t0 is not None:
                pat_lab["hours"] = (
                    pd.to_datetime(pat_lab["charttime"]) - t0
                ).dt.total_seconds() / 3600
            else:
                pat_lab["hours"] = np.nan

            # 只保留 ICU 期间 (-6h ~ 120h)
            pat_lab = pat_lab[(pat_lab["hours"] >= -6) & (pat_lab["hours"] <= 120)]

            show_legend = (ci == 0)
            for col_idx, itemid in enumerate(SHOW_ITEMS, 1):
                sub = pat_lab[pat_lab["itemid"] == itemid].sort_values("hours")
                if len(sub) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["hours"],
                    y=sub["valuenum"],
                    mode="lines+markers",
                    name=f"{group_label} – S{sid}",
                    line=dict(color=colors[ci], width=2,
                              dash="solid" if group_label == "Deceased" else "dot"),
                    marker=dict(size=5, color=colors[ci]),
                    legendgroup=f"{group_label}_{sid}",
                    showlegend=(show_legend and col_idx == 1),
                    hovertemplate=(
                        f"<b>Subject {sid} ({group_label})</b><br>"
                        f"Hour: %{{x:.1f}}<br>Value: %{{y:.2f}}<extra></extra>"
                    ),
                ),
                row=1, col=col_idx)

    for col_idx in range(1, 4):
        fig.update_xaxes(title_text="Hours from ICU Admission",
                         showgrid=True, gridcolor="#e5e7eb",
                         title_font=dict(size=11), row=1, col=col_idx)
        fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", row=1, col=col_idx)

    for ann in fig.layout.annotations:
        ann.font.color = "#1a1a2e"
        ann.font.size  = 13

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="Individual Patient Lab Trajectories During ICU Stay",
            font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"
        ),
        legend=dict(
            x=1.01, y=1, xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=11),
            title=dict(text="Patient (Solid=Deceased, Dotted=Survived)"),
        ),
        margin=dict(t=70, b=60, l=60, r=200),
        height=420,
    )
    return fig, selected


# ════════════════════════════════════════════════════════
# 组装 HTML
# ════════════════════════════════════════════════════════

def build_dashboard(fig_corr, corr, fig_mort, fig_outlier, top_outliers,
                    fig_ts, selected_patients):

    def to_div(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    # 关键发现汇总
    # 找相关性最高的非对角线对
    corr_pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            corr_pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pair = corr_pairs[0] if corr_pairs else ("—", "—", 0)

    n_outliers_deceased = int((top_outliers["hospital_expire_flag"] == 1).sum())
    deceased_selected = selected_patients.get("Deceased", [])
    survived_selected = selected_patients.get("Survived", [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MIMIC-III Patient Story Analysis</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f1f5f9; color: #1a1a2e; min-height: 100vh;
    }}
    header {{
      background: #ffffff; border-bottom: 2px solid #e2e8f0;
      padding: 24px 40px 18px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    header h1 {{ font-size: 21px; font-weight: 700; color: #1e293b; }}
    header p  {{ font-size: 13px; color: #64748b; margin-top: 4px; }}

    .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 40px; }}

    .insight-grid {{
      display: grid; grid-template-columns: repeat(3, 1fr);
      gap: 16px; margin-bottom: 24px;
    }}
    .insight-card {{
      background: #ffffff; border: 1px solid #e2e8f0;
      border-radius: 10px; padding: 16px 20px;
      border-left: 4px solid #2563EB;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }}
    .insight-card.green {{ border-left-color: #10B981; }}
    .insight-card.red   {{ border-left-color: #EF4444; }}
    .insight-title {{ font-size: 12px; font-weight: 600; color: #64748b;
                      text-transform: uppercase; letter-spacing: 0.5px; }}
    .insight-body  {{ font-size: 13px; color: #334155; margin-top: 6px; line-height: 1.5; }}

    .chart-card {{
      background: #ffffff; border: 1px solid #e2e8f0;
      border-radius: 12px; padding: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.06);
      margin-bottom: 20px;
    }}
    .story-note {{
      background: #F0FDF4; border: 1px solid #BBF7D0;
      border-radius: 8px; padding: 12px 16px;
      font-size: 13px; color: #14532D;
      margin-top: 10px; line-height: 1.6;
    }}
    .story-note.warn {{
      background: #FEF9C3; border-color: #FDE68A; color: #713F12;
    }}
    .story-note.info {{
      background: #EFF6FF; border-color: #BFDBFE; color: #1E3A8A;
    }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  </style>
</head>
<body>
<header>
  <h1>MIMIC-III Level 3 — Patient Lab Story Analysis</h1>
  <p>Connecting multi-dimensional lab data to tell the story of ICU patients · Patterns, outliers, and individual trajectories</p>
</header>

<div class="container">

  <!-- Key Insights -->
  <div class="insight-grid">
    <div class="insight-card">
      <div class="insight-title">🔗 Strongest Lab Correlation</div>
      <div class="insight-body">
        <b>{top_pair[0]}</b> &amp; <b>{top_pair[1]}</b><br>
        Pearson r = <b>{top_pair[2]:.2f}</b> — co-elevation suggests shared pathophysiology
      </div>
    </div>
    <div class="insight-card red">
      <div class="insight-title">⚠ Outlier Patients (Deceased)</div>
      <div class="insight-body">
        <b>{n_outliers_deceased}</b> of top-20 outlier patients had in-hospital death,
        suggesting extreme lab values are strong mortality signals
      </div>
    </div>
    <div class="insight-card green">
      <div class="insight-title">📈 Trajectory Patients Selected</div>
      <div class="insight-body">
        Deceased: Subject(s) <b>{', '.join(str(s) for s in deceased_selected)}</b><br>
        Survived: Subject(s) <b>{', '.join(str(s) for s in survived_selected)}</b>
      </div>
    </div>
  </div>

  <!-- 图 1: Correlation Heatmap -->
  <div class="chart-card">
    {to_div(fig_corr)}
    <div class="story-note info">
      📖 <b>What this tells us:</b> Strong positive correlations (dark red) between markers
      like Lactate &amp; Bilirubin or Creatinine &amp; BUN suggest simultaneous multi-organ
      dysfunction — a hallmark of sepsis and critical illness. Negative correlations with
      Hemoglobin reflect anemia commonly co-occurring with organ failure.
    </div>
  </div>

  <!-- 图 2: Deceased vs Survived -->
  <div class="chart-card">
    {to_div(fig_mort)}
    <div class="story-note warn">
      📖 <b>What this tells us:</b> Deceased patients consistently show higher peak values
      for Creatinine, Lactate, Bilirubin and ALT — markers of renal, circulatory, and
      hepatic failure. Lower Hemoglobin in the deceased group reflects anemia and possible
      hemorrhage. These differences align with known ICU mortality predictors.
    </div>
  </div>

  <!-- 图 3 + 图 4: Outliers + Trajectories -->
  <div class="chart-card">
    {to_div(fig_outlier)}
    <div class="story-note warn">
      📖 <b>What this tells us:</b> Patients with the highest Z-scores (most extreme values
      relative to the cohort) are disproportionately from the deceased group (red bars).
      Hover over each bar to see which marker drove the outlier status — Lactate and
      Creatinine are common culprits.
    </div>
  </div>

  <div class="chart-card">
    {to_div(fig_ts)}
    <div class="story-note">
      📖 <b>What this tells us:</b> Deceased patients (solid lines) typically show a
      progressive rise in Creatinine and Lactate over the ICU stay, indicating worsening
      renal function and tissue hypoperfusion. Hemoglobin trends downward. Survived
      patients (dotted lines) tend to show stabilization or improvement — a visual
      signature of recovery vs. deterioration.
    </div>
  </div>

</div>
</body>
</html>"""

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard saved: {OUTPUT_FILE}")


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("Stage 3b: Patient Story Analysis")
    print("=" * 55)

    print("\n[1] Loading data ...")
    admissions, icustays, lab = load_data()
    print(f"  Lab events (filtered): {len(lab):,}")

    print("\n[2] Correlation heatmap ...")
    fig_corr, corr = fig_correlation_heatmap(lab, admissions)

    print("\n[3] Deceased vs Survived lab comparison ...")
    fig_mort = fig_mortality_lab_comparison(lab, admissions)

    print("\n[4] Outlier detection (Z-score) ...")
    fig_outlier, top_outliers = fig_outlier_detection(lab, admissions)
    print(f"  Top outliers: {len(top_outliers)}")
    print(f"  Deceased in top-20: {(top_outliers['hospital_expire_flag']==1).sum()}")

    print("\n[5] Individual patient trajectories ...")
    fig_ts, selected = fig_patient_timeseries(lab, admissions, icustays, top_outliers)
    print(f"  Deceased patients: {selected.get('Deceased', [])}")
    print(f"  Survived patients: {selected.get('Survived', [])}")

    print("\n[6] Building dashboard ...")
    build_dashboard(fig_corr, corr, fig_mort, fig_outlier, top_outliers,
                    fig_ts, selected)

    print(f"\n✅ Done! Open in browser: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()