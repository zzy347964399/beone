"""
Stage 3a: eDISH — Evaluation of Drug-Induced Serious Hepatotoxicity
输入：LABEVENTS.csv
输出：edish_dashboard.html
运行：python stage3a_edish.py

eDISH 四象限定义（FDA 标准）：
  X 轴：Peak ALT / ULN  (ULN = 40 U/L)
  Y 轴：Peak Total Bilirubin / ULN  (ULN = 1.2 mg/dL)

  右上（Hy's Law）：ALT/ULN > 3  AND  Bili/ULN > 2  → 潜在严重肝毒性
  左上（Temple's Corollary）：ALT/ULN ≤ 3  AND  Bili/ULN > 2  → 胆汁淤积为主
  右下（Cholestasis/Hepatocellular）：ALT/ULN > 3  AND  Bili/ULN ≤ 2  → 肝细胞损伤为主
  左下（Normal）：ALT/ULN ≤ 3  AND  Bili/ULN ≤ 2  → 正常范围
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import os

DATA_DIR = "/Users/pickle/.cache/kagglehub/datasets/asjad99/mimiciii/versions/1/mimic-iii-clinical-database-demo-1.4"
OUTPUT_FILE = "edish_dashboard.html"

# Lab itemids
ITEMID_ALT      = 50861   # ALT (SGPT)        U/L
ITEMID_BILI     = 50885   # Total Bilirubin    mg/dL
ITEMID_CREAT    = 50912   # Creatinine         mg/dL
ITEMID_WBC      = 51301   # WBC                K/uL
ITEMID_HGB      = 51222   # Hemoglobin         g/dL
ITEMID_PLT      = 51265   # Platelets          K/uL

# Upper Limits of Normal
ULN_ALT  = 40.0    # U/L
ULN_BILI = 1.2     # mg/dL

# Hy's Law thresholds (× ULN)
HYS_ALT_X  = 3.0
HYS_BILI_X = 2.0

LAYOUT_BASE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#1a1a2e", family="Arial, sans-serif"),
)


# ════════════════════════════════════════════════════════
# 数据提取
# ════════════════════════════════════════════════════════

def load_labevents():
    path = os.path.join(DATA_DIR, "LABEVENTS.csv")
    df = pd.read_csv(path,
                     usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
                     parse_dates=["charttime"],
                     low_memory=False)
    # 排除无效值
    df = df[df["valuenum"].notna() & (df["valuenum"] > 0)]
    return df


def extract_peak(labevents, itemid):
    """按 subject_id 取某指标的峰值。"""
    sub = labevents[labevents["itemid"] == itemid]
    peak = sub.groupby("subject_id")["valuenum"].max().reset_index()
    peak.columns = ["subject_id", f"peak_{itemid}"]
    return peak


def build_edish_df(labevents):
    alt_peak  = extract_peak(labevents, ITEMID_ALT)
    bili_peak = extract_peak(labevents, ITEMID_BILI)

    df = alt_peak.merge(bili_peak, on="subject_id", how="inner")
    df = df.rename(columns={
        f"peak_{ITEMID_ALT}":  "alt_peak",
        f"peak_{ITEMID_BILI}": "bili_peak",
    })

    # 归一化
    df["alt_uln"]  = df["alt_peak"]  / ULN_ALT
    df["bili_uln"] = df["bili_peak"] / ULN_BILI

    # 象限分类
    def classify(row):
        a = row["alt_uln"] > HYS_ALT_X
        b = row["bili_uln"] > HYS_BILI_X
        if a and b:     return "Hy's Law"
        if b and not a: return "Temple's Corollary"
        if a and not b: return "Hepatocellular"
        return "Normal"

    df["quadrant"] = df.apply(classify, axis=1)
    df["hys_law"]  = df["quadrant"] == "Hy's Law"

    print(f"  患者数（有 ALT + Bili 记录）：{len(df)}")
    print(f"  Hy's Law 患者：{df['hys_law'].sum()}")
    print(df["quadrant"].value_counts().to_string())
    return df


# ════════════════════════════════════════════════════════
# 延伸：KDIGO 肾脏分期（从肌酐）
# ════════════════════════════════════════════════════════

def build_kdigo(labevents):
    """
    简化版 KDIGO：对每位患者，以该患者肌酐最小值作为基线，
    计算峰值肌酐相对基线的倍数和绝对升高量。
    Stage 1: ≥0.3 mg/dL 升高 or 1.5–1.9× baseline
    Stage 2: 2.0–2.9× baseline
    Stage 3: ≥3.0× baseline or ≥4.0 mg/dL
    """
    creat = labevents[labevents["itemid"] == ITEMID_CREAT].copy()
    creat = creat[(creat["valuenum"] > 0) & (creat["valuenum"] < 30)]

    stats = creat.groupby("subject_id")["valuenum"].agg(
        creat_baseline="min",
        creat_peak="max",
    ).reset_index()
    stats["creat_ratio"]  = stats["creat_peak"] / stats["creat_baseline"]
    stats["creat_delta"]  = stats["creat_peak"] - stats["creat_baseline"]

    def kdigo_stage(row):
        r = row["creat_ratio"]
        d = row["creat_delta"]
        p = row["creat_peak"]
        if r >= 3.0 or p >= 4.0:          return 3
        if 2.0 <= r < 3.0:                return 2
        if (1.5 <= r < 2.0) or d >= 0.3:  return 1
        return 0

    stats["kdigo_stage"] = stats.apply(kdigo_stage, axis=1)
    print("\n  KDIGO 分期分布：")
    print(stats["kdigo_stage"].value_counts().sort_index().to_string())
    return stats


# ════════════════════════════════════════════════════════
# 延伸：CTCAE 血液毒性分级
# ════════════════════════════════════════════════════════

def build_ctcae(labevents):
    """
    基于最差值（最低值对 WBC/HGB/PLT）进行 CTCAE 分级。
    WBC:  Grade 3 < 2.0, Grade 4 < 1.0  (K/uL)
    HGB:  Grade 3 < 8.0, Grade 4 危及生命（用 < 6.5 近似）(g/dL)
    PLT:  Grade 3 < 50,  Grade 4 < 25   (K/uL)
    """
    CTCAE_ITEMS = {
        ITEMID_WBC: ("wbc",  [(4, 2.0), (3, 1.0)]),   # (grade, threshold)
        ITEMID_HGB: ("hgb",  [(4, 8.0), (3, 6.5)]),
        ITEMID_PLT: ("plt",  [(4, 50),  (3, 25)]),
    }

    results = []
    for itemid, (name, grades) in CTCAE_ITEMS.items():
        sub = labevents[labevents["itemid"] == itemid].copy()
        sub = sub[sub["valuenum"] > 0]
        nadir = sub.groupby("subject_id")["valuenum"].min().reset_index()
        nadir.columns = ["subject_id", f"{name}_nadir"]

        def grade(val):
            for g, thresh in grades:
                if val < thresh: return g
            return 0 if val > 0 else np.nan

        nadir[f"{name}_ctcae"] = nadir[f"{name}_nadir"].apply(grade)
        results.append(nadir.set_index("subject_id"))

    ctcae_df = pd.concat(results, axis=1).reset_index()
    print("\n  CTCAE 分级汇总（最高分级人数）：")
    for col in [c for c in ctcae_df.columns if "_ctcae" in c]:
        dist = ctcae_df[col].value_counts().sort_index()
        print(f"    {col}: {dist.to_dict()}")
    return ctcae_df


# ════════════════════════════════════════════════════════
# 图表
# ════════════════════════════════════════════════════════

def fig_edish(edish_df, kdigo_df, ctcae_df):
    """主 eDISH 散点图 + 象限标注。"""

    # 合并 KDIGO 和 CTCAE 用于 hover
    df = edish_df.merge(kdigo_df[["subject_id","kdigo_stage","creat_peak","creat_ratio"]],
                        on="subject_id", how="left")
    df = df.merge(ctcae_df, on="subject_id", how="left")

    # 颜色分组
    QUAD_STYLE = {
        "Normal":              dict(color="#94A3B8", symbol="circle",       size=9,  opacity=0.65),
        "Hepatocellular":      dict(color="#F59E0B", symbol="diamond",      size=11, opacity=0.80),
        "Temple's Corollary":  dict(color="#8B5CF6", symbol="square",       size=11, opacity=0.80),
        "Hy's Law":            dict(color="#EF4444", symbol="star",         size=16, opacity=1.00),
    }

    fig = go.Figure()

    for quad, style in QUAD_STYLE.items():
        sub = df[df["quadrant"] == quad]
        if len(sub) == 0:
            continue

        hover = (
            "<b>Subject %{customdata[0]}</b><br>"
            "Peak ALT: %{customdata[1]:.1f} U/L (%{x:.2f}× ULN)<br>"
            "Peak Bili: %{customdata[2]:.2f} mg/dL (%{y:.2f}× ULN)<br>"
            "KDIGO Stage: %{customdata[3]}<br>"
            "WBC CTCAE: %{customdata[4]}<br>"
            "HGB CTCAE: %{customdata[5]}<br>"
            "PLT CTCAE: %{customdata[6]}"
            "<extra></extra>"
        )
        customdata = sub[[
            "subject_id","alt_peak","bili_peak",
            "kdigo_stage","wbc_ctcae","hgb_ctcae","plt_ctcae"
        ]].values

        fig.add_trace(go.Scatter(
            x=sub["alt_uln"],
            y=sub["bili_uln"],
            mode="markers",
            name=f"{quad} (n={len(sub)})",
            marker=dict(
                color=style["color"],
                symbol=style["symbol"],
                size=style["size"],
                opacity=style["opacity"],
                line=dict(color="#ffffff", width=0.8),
            ),
            customdata=customdata,
            hovertemplate=hover,
        ))

    # 参考线
    ref_lines = [
        (True,  1.0, "#94A3B8", "dash",      "ALT = 1× ULN"),
        (True,  3.0, "#EF4444", "dashdot",   "ALT = 3× ULN (Hy's threshold)"),
        (False, 1.0, "#94A3B8", "dash",      "Bili = 1× ULN"),
        (False, 2.0, "#EF4444", "dashdot",   "Bili = 2× ULN (Hy's threshold)"),
    ]
    for is_vertical, val, color, dash, label in ref_lines:
        if is_vertical:
            fig.add_vline(x=val, line_color=color, line_dash=dash,
                          line_width=1.2, opacity=0.6)
        else:
            fig.add_hline(y=val, line_color=color, line_dash=dash,
                          line_width=1.2, opacity=0.6)

    # 象限标签
    quad_labels = [
        (0.3,  3.5,  "Temple's Corollary",  "#8B5CF6"),
        (4.0,  3.5,  "Hy's Law",            "#EF4444"),
        (0.3,  0.3,  "Normal",              "#64748B"),
        (4.0,  0.3,  "Hepatocellular",      "#F59E0B"),
    ]
    for x, y, text, color in quad_labels:
        fig.add_annotation(
            x=np.log10(x), y=np.log10(y),
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=11, color=color),
            xref="x", yref="y",
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=3,
        )

    # 坐标轴：对数刻度，手动设置 tickvals
    tick_vals  = [0.1, 0.3, 1, 3, 10, 30, 100]
    tick_texts = ["0.1×", "0.3×", "1×", "3×", "10×", "30×", "100×"]

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(
            text="eDISH Plot — Evaluation of Drug-Induced Serious Hepatotoxicity",
            font=dict(size=17, color="#1a1a2e"), x=0, xanchor="left"
        ),
        xaxis=dict(
            title="Peak ALT / ULN  (ULN = 40 U/L)",
            type="log",
            tickvals=tick_vals,
            ticktext=tick_texts,
            showgrid=True, gridcolor="#e5e7eb",
            title_font=dict(size=13),
            range=[np.log10(0.05), np.log10(200)],
        ),
        yaxis=dict(
            title="Peak Total Bilirubin / ULN  (ULN = 1.2 mg/dL)",
            type="log",
            tickvals=tick_vals,
            ticktext=tick_texts,
            showgrid=True, gridcolor="#e5e7eb",
            title_font=dict(size=13),
            range=[np.log10(0.05), np.log10(50)],
        ),
        legend=dict(
            x=1.01, y=1, xanchor="left",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e2e8f0", borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(t=60, b=60, l=70, r=180),
        height=560,
        hovermode="closest",
    )
    return fig


def fig_kdigo_bar(kdigo_df):
    """KDIGO 分期柱状图。"""
    counts = kdigo_df["kdigo_stage"].value_counts().sort_index()
    labels = {0: "No AKI", 1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}
    colors = {0: "#94A3B8", 1: "#FCD34D", 2: "#F97316", 3: "#EF4444"}

    x = [labels.get(i, str(i)) for i in counts.index]
    c = [colors.get(i, "#888") for i in counts.index]

    fig = go.Figure(go.Bar(
        x=x, y=counts.values,
        marker_color=c,
        marker_line=dict(color="#ffffff", width=1),
        text=counts.values, textposition="outside",
        hovertemplate="<b>%{x}</b><br>Patients: %{y}<extra></extra>",
    ))
    # X 轴标签补充分期标准（用 <br> 换行，Plotly 支持 HTML 标签）
    x_labels = {
        "No AKI":  "No AKI<br>(< 1.5× baseline, Δ < 0.3 mg/dL)",
        "Stage 1": "Stage 1<br>(1.5–1.9× or Δ ≥ 0.3 mg/dL)",
        "Stage 2": "Stage 2<br>(2.0–2.9× baseline)",
        "Stage 3": "Stage 3<br>(≥ 3× or ≥ 4.0 mg/dL)",
    }
    x_display = [x_labels.get(v, v) for v in x]

    fig.update_traces(x=x_display)
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Renal Safety — KDIGO AKI Staging (from Creatinine)",
                   font=dict(size=15, color="#1a1a2e"), x=0, xanchor="left"),
        yaxis=dict(title="Number of Patients", showgrid=True, gridcolor="#e5e7eb",
                   range=[0, counts.max() * 1.3], dtick=1,
                   title_font=dict(size=12)),
        xaxis=dict(title="KDIGO Stage (creatinine-based criteria)",
                   title_font=dict(size=12),
                   tickfont=dict(size=10)),
        margin=dict(t=50, b=80, l=60, r=20),
        height=360,
        showlegend=False,
    )
    return fig


def fig_ctcae_bar(ctcae_df):
    """CTCAE 血液毒性分级堆叠柱状图。"""
    metrics = [
        ("wbc_ctcae", "WBC",        "#2563EB"),
        ("hgb_ctcae", "Hemoglobin", "#10B981"),
        ("plt_ctcae", "Platelets",  "#F59E0B"),
    ]
    grade_labels = {0: "G0 (Normal)", 1: "G1", 2: "G2", 3: "G3", 4: "G4"}
    grade_colors = {
        0: "#E2E8F0", 1: "#FEF9C3", 2: "#FDE68A", 3: "#F97316", 4: "#EF4444"
    }

    all_grades = sorted(ctcae_df[[m[0] for m in metrics]].stack().dropna().unique())
    fig = go.Figure()

    for grade in all_grades:
        counts = []
        for col, label, _ in metrics:
            counts.append((ctcae_df[col] == grade).sum())
        fig.add_trace(go.Bar(
            name=grade_labels.get(int(grade), f"G{int(grade)}"),
            x=[m[1] for m in metrics],
            y=counts,
            marker_color=grade_colors.get(int(grade), "#888"),
            marker_line=dict(color="#ffffff", width=0.8),
            hovertemplate="<b>%{x}</b><br>Grade %{name}<br>Patients: %{y}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Hematological Toxicity — CTCAE Grading",
                   font=dict(size=15, color="#1a1a2e"), x=0, xanchor="left"),
        barmode="stack",
        yaxis=dict(title="Number of Patients", showgrid=True, gridcolor="#e5e7eb"),
        xaxis=dict(title="Lab Marker"),
        legend=dict(title="CTCAE Grade", font=dict(size=11)),
        margin=dict(t=50, b=50, l=60, r=120),
        height=340,
    )
    return fig


# ════════════════════════════════════════════════════════
# 组装 HTML
# ════════════════════════════════════════════════════════

def build_dashboard(fig_main, fig_kdigo, fig_ctcae, edish_df):
    n_hys = int(edish_df["hys_law"].sum())
    n_total = len(edish_df)
    hys_ids = edish_df[edish_df["hys_law"]]["subject_id"].tolist()
    hys_list = ", ".join(str(i) for i in sorted(hys_ids)) if hys_ids else "None"

    def to_div(fig):
        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MIMIC-III eDISH Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f1f5f9;
      color: #1a1a2e;
      min-height: 100vh;
    }}
    header {{
      background: #ffffff;
      border-bottom: 2px solid #e2e8f0;
      padding: 24px 40px 18px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    header h1 {{ font-size: 21px; font-weight: 700; color: #1e293b; }}
    header p  {{ font-size: 13px; color: #64748b; margin-top: 4px; }}

    .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 40px; }}

    .summary-bar {{
      display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px;
    }}
    .stat-card {{
      background: #ffffff; border: 1px solid #e2e8f0;
      border-radius: 10px; padding: 16px 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.05);
      min-width: 160px;
    }}
    .stat-card.danger {{ border-left: 4px solid #EF4444; }}
    .stat-card.warn   {{ border-left: 4px solid #F59E0B; }}
    .stat-card.info   {{ border-left: 4px solid #2563EB; }}
    .stat-val  {{ font-size: 28px; font-weight: 700; color: #1e293b; }}
    .stat-label{{ font-size: 12px; color: #64748b; margin-top: 2px; }}

    .hys-box {{
      background: #FEF2F2; border: 1px solid #FECACA;
      border-radius: 10px; padding: 14px 20px;
      margin-bottom: 24px; font-size: 13px; color: #7F1D1D;
    }}
    .hys-box b {{ color: #DC2626; }}

    .chart-card {{
      background: #ffffff; border: 1px solid #e2e8f0;
      border-radius: 12px; padding: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.06);
      margin-bottom: 20px;
    }}
    .grid-2 {{
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 20px; margin-bottom: 20px;
    }}

    .legend-table {{
      width: 100%; border-collapse: collapse;
      font-size: 13px; margin-top: 8px;
    }}
    .legend-table th {{
      background: #f8fafc; text-align: left;
      padding: 8px 12px; border-bottom: 2px solid #e2e8f0;
      color: #475569; font-weight: 600;
    }}
    .legend-table td {{
      padding: 8px 12px; border-bottom: 1px solid #f1f5f9;
      color: #334155;
    }}
    .dot {{
      display: inline-block; width: 10px; height: 10px;
      border-radius: 50%; margin-right: 6px; vertical-align: middle;
    }}
  </style>
</head>
<body>
<header>
  <h1>MIMIC-III Level 3 — eDISH Pharmacovigilance Dashboard</h1>
  <p>Evaluation of Drug-Induced Serious Hepatotoxicity · Renal Safety (KDIGO) · Hematological Toxicity (CTCAE)</p>
</header>

<div class="container">

  <!-- 统计卡片 -->
  <div class="summary-bar">
    <div class="stat-card info">
      <div class="stat-val">{n_total}</div>
      <div class="stat-label">Patients with ALT + Bili Data</div>
    </div>
    <div class="stat-card danger">
      <div class="stat-val">{n_hys}</div>
      <div class="stat-label">Hy's Law Patients</div>
    </div>
    <div class="stat-card warn">
      <div class="stat-val">{n_hys/n_total*100:.1f}%</div>
      <div class="stat-label">Hy's Law Rate</div>
    </div>
  </div>

  <!-- Hy's Law 患者列表 -->
  <div class="hys-box">
    <b>⚠ Hy's Law Patients (ALT &gt; 3× ULN AND Bili &gt; 2× ULN):</b>
    &nbsp;Subject IDs: {hys_list}
  </div>

  <!-- 主 eDISH 图 -->
  <div class="chart-card">
    {to_div(fig_main)}
  </div>

  <!-- 象限说明表 -->
  <div class="chart-card">
    <b style="font-size:14px;">eDISH Quadrant Definitions</b>
    <table class="legend-table">
      <tr>
        <th>Quadrant</th><th>ALT / ULN</th><th>Bili / ULN</th>
        <th>Interpretation</th>
      </tr>
      <tr>
        <td><span class="dot" style="background:#EF4444"></span><b>Hy's Law</b></td>
        <td>&gt; 3×</td><td>&gt; 2×</td>
        <td>Potential serious hepatotoxicity — highest risk signal</td>
      </tr>
      <tr>
        <td><span class="dot" style="background:#8B5CF6"></span>Temple's Corollary</td>
        <td>≤ 3×</td><td>&gt; 2×</td>
        <td>Predominantly cholestatic — bile duct dysfunction</td>
      </tr>
      <tr>
        <td><span class="dot" style="background:#F59E0B"></span>Hepatocellular</td>
        <td>&gt; 3×</td><td>≤ 2×</td>
        <td>Hepatocellular injury without jaundice</td>
      </tr>
      <tr>
        <td><span class="dot" style="background:#94A3B8"></span>Normal</td>
        <td>≤ 3×</td><td>≤ 2×</td>
        <td>Within normal / non-concerning range</td>
      </tr>
    </table>
  </div>

  <!-- KDIGO + CTCAE -->
  <div class="grid-2">
    <div class="chart-card">{to_div(fig_kdigo)}</div>
    <div class="chart-card">{to_div(fig_ctcae)}</div>
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
    print("Stage 3a: eDISH Pharmacovigilance")
    print("=" * 55)

    print("\n[1] Loading LABEVENTS ...")
    labevents = load_labevents()
    print(f"  Rows: {len(labevents):,}")

    print("\n[2] Building eDISH dataset ...")
    edish_df = build_edish_df(labevents)

    print("\n[3] KDIGO renal staging ...")
    kdigo_df = build_kdigo(labevents)

    print("\n[4] CTCAE hematological grading ...")
    ctcae_df = build_ctcae(labevents)

    print("\n[5] Generating charts ...")
    fig_main  = fig_edish(edish_df, kdigo_df, ctcae_df)
    fig_kdigo = fig_kdigo_bar(kdigo_df)
    fig_ctcae = fig_ctcae_bar(ctcae_df)

    print("\n[6] Building dashboard ...")
    build_dashboard(fig_main, fig_kdigo, fig_ctcae, edish_df)

    print(f"\n✅ Done! Open in browser: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()