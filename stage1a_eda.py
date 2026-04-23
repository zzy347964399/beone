"""
阶段 1a：EDA 探索性仪表板
运行：python stage1a_eda.py
输出：eda_dashboard.html
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os

DATA_DIR = "/Users/pickle/.cache/kagglehub/datasets/asjad99/mimiciii/versions/1/mimic-iii-clinical-database-demo-1.4"
OUTPUT_FILE = "eda_dashboard.html"

# 全局白底主题
LAYOUT_BASE = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8f9fa",
    font=dict(color="#1a1a2e", family="Arial, sans-serif"),
)

ACCENT  = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"]
C_DEAD  = "#EF4444"
C_ALIVE = "#10B981"

def hex_to_rgba(hex_color, alpha=0.25):
    """Convert #RRGGBB to rgba(r,g,b,alpha) for Plotly fillcolor."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


# ════════════════════════════════════════════════════════
# 数据加载
# ════════════════════════════════════════════════════════

def load_data():
    def read(fname, cols=None, dates=None):
        path = os.path.join(DATA_DIR, fname)
        return pd.read_csv(path, usecols=cols, parse_dates=dates, low_memory=False)

    admissions = read("ADMISSIONS.csv",
        cols=["subject_id","hadm_id","admittime","dischtime",
              "admission_type","hospital_expire_flag"],
        dates=["admittime","dischtime"])
    icustays = read("ICUSTAYS.csv",
        cols=["subject_id","hadm_id","icustay_id","first_careunit","los"])
    diagnoses = read("DIAGNOSES_ICD.csv",
        cols=["subject_id","hadm_id","icd9_code"])
    d_icd = read("D_ICD_DIAGNOSES.csv",
        cols=["icd9_code","long_title"])
    labevents = read("LABEVENTS.csv",
        cols=["subject_id","hadm_id","itemid","charttime","valuenum"],
        dates=["charttime"])

    admissions["hospital_expire_flag"] = (
        admissions["hospital_expire_flag"].fillna(0).astype(int))
    return admissions, icustays, diagnoses, d_icd, labevents


# ════════════════════════════════════════════════════════
# 图 1：In-Hospital Mortality Rate
# ════════════════════════════════════════════════════════

def fig_mortality(admissions):
    n_total = len(admissions)
    n_dead  = int(admissions["hospital_expire_flag"].sum())
    n_alive = n_total - n_dead
    rate    = n_dead / n_total * 100

    fig = go.Figure(go.Pie(
        labels=["Survived", "In-Hospital Death"],
        values=[n_alive, n_dead],
        hole=0.58,
        marker=dict(colors=[C_ALIVE, C_DEAD],
                    line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=13, color="#1a1a2e"),
        hovertemplate="%{label}: %{value} patients (%{percent})<extra></extra>",
        direction="clockwise",
    ))
    # 中心标注
    fig.add_annotation(
        text=f"<b>{rate:.1f}%</b><br><span style='font-size:11px'>Mortality</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color="#1a1a2e"),
        xanchor="center", yanchor="middle",
    )
    # 底部统计行 — 用两行分开，行距充足
    fig.add_annotation(
        text=f"Total Admissions: <b>{n_total}</b>",
        x=0.5, y=-0.10, showarrow=False,
        font=dict(size=12, color="#444"),
        xref="paper", yref="paper", xanchor="center",
    )
    fig.add_annotation(
        text=f"Deaths: <b>{n_dead}</b> &nbsp;|&nbsp; Survived: <b>{n_alive}</b>",
        x=0.5, y=-0.18, showarrow=False,
        font=dict(size=12, color="#444"),
        xref="paper", yref="paper", xanchor="center",
    )
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="In-Hospital Mortality Rate",
                   font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"),
        showlegend=True,
        legend=dict(orientation="h", y=-0.28, x=0.5, xanchor="center",
                    font=dict(size=12)),
        margin=dict(t=50, b=110, l=20, r=20),
        height=400,
    )
    return fig


# ════════════════════════════════════════════════════════
# 图 2：Top 10 Most Common Diagnoses
# ════════════════════════════════════════════════════════

def fig_top_diagnoses(diagnoses, d_icd):
    merged = diagnoses.merge(d_icd, on="icd9_code", how="left")
    merged["label"] = merged["long_title"].fillna(merged["icd9_code"].astype(str))

    top10 = (
        merged.groupby("label")["hadm_id"].count()
        .nlargest(10).reset_index()
        .rename(columns={"hadm_id": "count"})
        .sort_values("count")
    )
    def wrap_label(text, width=30):
        words = text.split()
        lines, cur = [], []
        for w in words:
            if sum(len(x)+1 for x in cur) + len(w) > width and cur:
                lines.append(' '.join(cur))
                cur = [w]
            else:
                cur.append(w)
        if cur: lines.append(' '.join(cur))
        return '<br>'.join(lines)
    top10["label_short"] = top10["label"].apply(wrap_label)

    fig = go.Figure(go.Bar(
        x=top10["count"],
        y=top10["label_short"],
        orientation="h",
        marker=dict(
            color=top10["count"],
            colorscale=[[0,"#93C5FD"],[1,"#1D4ED8"]],
            showscale=False,
        ),
        text=top10["count"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Top 10 Most Common Diagnoses (ICD-9)",
                   font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"),
        xaxis=dict(title="Count", title_font=dict(size=12),
                   showgrid=True, gridcolor="#e5e7eb"),
        yaxis=dict(automargin=True, tickfont=dict(size=10)),
        margin=dict(t=50, b=60, l=20, r=60),   # r=60 给 text 留空间
        height=420,
    )
    return fig


# ════════════════════════════════════════════════════════
# 图 3：ICU Length of Stay by Care Unit
# ════════════════════════════════════════════════════════

def fig_los_by_careunit(icustays):
    df = icustays[icustays["los"] > 0].copy()
    units = df["first_careunit"].value_counts().index.tolist()
    y_max = min(df["los"].quantile(0.95) * 1.2, 35)

    fig = go.Figure()
    for i, unit in enumerate(units):
        sub = df[df["first_careunit"] == unit]["los"]
        fig.add_trace(go.Box(
            y=sub,
            name=unit,
            marker=dict(color=ACCENT[i % len(ACCENT)]),
            line=dict(color=ACCENT[i % len(ACCENT)]),
            fillcolor=hex_to_rgba(ACCENT[i % len(ACCENT)], 0.25),
            boxmean="sd",
            hovertemplate=f"<b>{unit}</b><br>LOS: %{{y:.1f}} days<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#1a1a2e", family="Arial, sans-serif"),
        template="none",
        title=dict(text="ICU Length of Stay by Care Unit",
                   font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"),
        yaxis=dict(
            title="LOS (days)",
            range=[0, y_max],
            showgrid=True, gridcolor="#e5e7eb",
            zeroline=True, zerolinecolor="#cbd5e1",
            title_font=dict(size=12),
            tickfont=dict(size=11),
            showline=True, linecolor="#e5e7eb",
        ),
        xaxis=dict(
            title="Care Unit",
            title_font=dict(size=12),
            tickfont=dict(size=11),
            showline=True, linecolor="#e5e7eb",
        ),
        showlegend=False,
        margin=dict(t=55, b=60, l=65, r=20),
        height=440,
        boxgap=0.3,
        boxgroupgap=0.2,
    )
    return fig


# ════════════════════════════════════════════════════════
# 图 4：Mortality Rate by Admission Type
# ════════════════════════════════════════════════════════

def fig_mortality_by_admission_type(admissions):
    grouped = (
        admissions.groupby("admission_type")["hospital_expire_flag"]
        .agg(total="count", dead="sum").reset_index()
    )
    grouped["rate"] = grouped["dead"] / grouped["total"] * 100

    COLOR_MAP = {"EMERGENCY": "#EF4444", "ELECTIVE": "#10B981", "URGENT": "#F59E0B"}
    colors = [COLOR_MAP.get(t, "#6B7280") for t in grouped["admission_type"]]

    fig = go.Figure(go.Bar(
        x=grouped["admission_type"],
        y=grouped["rate"],
        marker_color=colors,
        marker_line=dict(color="#ffffff", width=1.5),
        text=[f"{r:.1f}%" for r in grouped["rate"]],
        textposition="outside",
        textfont=dict(size=13, color="#1a1a2e"),
        width=0.45,       # 控制柱宽，不让柱子撑满
        customdata=grouped[["dead","total"]].values,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Mortality Rate: %{y:.1f}%<br>"
            "Deaths: %{customdata[0]} / Total: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    y_max = max(grouped["rate"].max() * 1.35, 10)   # 比最高值多 35%，保证标签不截断
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Mortality Rate by Admission Type",
                   font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"),
        yaxis=dict(title="Mortality Rate (%)", range=[0, y_max],
                   showgrid=True, gridcolor="#e5e7eb",
                   title_font=dict(size=12)),
        xaxis=dict(title="Admission Type", title_font=dict(size=12),
                   tickfont=dict(size=12)),
        bargap=0.5,
        margin=dict(t=50, b=60, l=60, r=20),
        height=400,
    )
    return fig


# ════════════════════════════════════════════════════════
# 图 5：Lab Value Distributions
# ════════════════════════════════════════════════════════

def fig_lab_distributions(labevents):
    LAB_ITEMS = {
        50912: ("Creatinine", "mg/dL", (0, 15),  1.0),
        51301: ("WBC",        "K/uL",  (0, 50),  2.0),
        51222: ("Hemoglobin", "g/dL",  (0, 20),  0.5),
    }
    COLORS = ["#2563EB", "#EF4444", "#10B981"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[v[0] for v in LAB_ITEMS.values()],
        horizontal_spacing=0.12,
    )

    for col_idx, (itemid, (name, unit, clip_range, bin_size)) in enumerate(LAB_ITEMS.items(), 1):
        sub = labevents[labevents["itemid"] == itemid]["valuenum"].dropna()
        sub = sub[(sub > clip_range[0]) & (sub <= clip_range[1])]

        print(f"  {name}: {len(sub)} records")
        if len(sub) == 0:
            fig.add_trace(go.Scatter(x=[], y=[], showlegend=False), row=1, col=col_idx)
            continue

        median_val = sub.median()
        mean_val   = sub.mean()

        fig.add_trace(
            go.Histogram(
                x=sub,
                xbins=dict(start=clip_range[0], end=clip_range[1], size=bin_size),
                marker=dict(
                    color=hex_to_rgba(COLORS[col_idx-1], 0.75),
                    line=dict(color=COLORS[col_idx-1], width=0.8),
                ),
                name=name,
                hovertemplate=f"<b>{name}</b><br>{unit}: %{{x:.1f}}<br>Count: %{{y}}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )
        # 中位线 + 均值线
        fig.add_vline(x=median_val, line_color=COLORS[col_idx-1], line_width=2,
                      line_dash="dash", row=1, col=col_idx,
                      annotation_text=f"Median {median_val:.1f}",
                      annotation_font=dict(size=10, color=COLORS[col_idx-1]),
                      annotation_position="top right")
        fig.add_vline(x=mean_val, line_color="#F59E0B", line_width=1.5,
                      line_dash="dot", row=1, col=col_idx)

        fig.update_xaxes(title_text=unit, row=1, col=col_idx,
                         title_font=dict(size=11),
                         showgrid=True, gridcolor="#e5e7eb",
                         showline=True, linecolor="#e5e7eb")
        fig.update_yaxes(title_text="Count" if col_idx == 1 else "",
                         row=1, col=col_idx,
                         showgrid=True, gridcolor="#e5e7eb")

    for ann in fig.layout.annotations:
        ann.font.color = "#1a1a2e"
        ann.font.size  = 13

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Lab Value Distributions — All Measurements",
                   font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"),
        margin=dict(t=80, b=60, l=60, r=20),
        height=420,
    )
    return fig


# ════════════════════════════════════════════════════════
# 图 6：ICU LOS Distribution
# ════════════════════════════════════════════════════════

def fig_los_distribution(icustays):
    los = icustays[icustays["los"] > 0]["los"]
    median_los = los.median()
    mean_los   = los.mean()
    p75_los    = los.quantile(0.75)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=los,
        xbins=dict(start=0, end=los.quantile(0.97), size=0.5),
        marker=dict(color="#8B5CF6", opacity=0.75,
                    line=dict(color="#ffffff", width=0.5)),
        name="ICU LOS",
        hovertemplate="LOS: %{x:.1f} days<br>Count: %{y}<extra></extra>",
    ))
    # 中位数线
    fig.add_vline(x=median_los, line_dash="dash", line_color="#EF4444", line_width=2,
                  annotation_text=f"Median: {median_los:.1f}d",
                  annotation_position="top right",
                  annotation_font=dict(size=11, color="#EF4444"))
    # 均值线
    fig.add_vline(x=mean_los, line_dash="dot", line_color="#F59E0B", line_width=2,
                  annotation_text=f"Mean: {mean_los:.1f}d",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color="#F59E0B"))
    # P75 线
    fig.add_vline(x=p75_los, line_dash="longdash", line_color="#2563EB", line_width=1.5,
                  annotation_text=f"P75: {p75_los:.1f}d",
                  annotation_position="top right",
                  annotation_font=dict(size=11, color="#2563EB"))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="ICU Length of Stay Distribution",
                   font=dict(size=16, color="#1a1a2e"), x=0, xanchor="left"),
        xaxis=dict(title="LOS (days)", showgrid=True, gridcolor="#e5e7eb",
                   title_font=dict(size=12)),
        yaxis=dict(title="Count", showgrid=True, gridcolor="#e5e7eb",
                   title_font=dict(size=12)),
        margin=dict(t=50, b=60, l=60, r=20),
        height=400,
        showlegend=False,
    )
    return fig


# ════════════════════════════════════════════════════════
# 组装 HTML
# ════════════════════════════════════════════════════════

def build_dashboard(figs, output_path):
    divs = []
    for fig in figs.values():
        div = pio.to_html(fig, full_html=False, include_plotlyjs=False)
        divs.append(f'<div class="chart-card">{div}</div>')

    charts_html = "\n".join(divs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MIMIC-III EDA Dashboard</title>
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
    header h1 {{
      font-size: 22px;
      font-weight: 700;
      color: #1e293b;
      letter-spacing: 0.3px;
    }}
    header p {{
      font-size: 13px;
      color: #64748b;
      margin-top: 4px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      padding: 28px 40px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    .chart-card {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 20px;
      box-shadow: 0 1px 6px rgba(0,0,0,0.06);
      transition: box-shadow 0.2s, border-color 0.2s;
      overflow: hidden;
    }}
    .chart-card:hover {{
      box-shadow: 0 4px 16px rgba(37,99,235,0.10);
      border-color: #93C5FD;
    }}
  </style>
</head>
<body>
  <header>
    <h1>MIMIC-III Clinical Data — EDA Dashboard</h1>
    <p>Level 1a · Exploratory Data Analysis · Beth Israel Deaconess Medical Center ICU (De-identified Demo Subset)</p>
  </header>
  <div class="grid">
    {charts_html}
  </div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Dashboard saved: {output_path}")


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("Stage 1a: EDA Dashboard")
    print("=" * 55)

    print("\n[1] Loading data ...")
    admissions, icustays, diagnoses, d_icd, labevents = load_data()
    print(f"  ADMISSIONS:  {len(admissions):,}")
    print(f"  ICUSTAYS:    {len(icustays):,}")
    print(f"  DIAGNOSES:   {len(diagnoses):,}")
    print(f"  LABEVENTS:   {len(labevents):,}")

    print("\n[2] Generating charts ...")
    figs = {
        "mortality":         fig_mortality(admissions),
        "top_diagnoses":     fig_top_diagnoses(diagnoses, d_icd),
        "los_by_careunit":   fig_los_by_careunit(icustays),
        "mortality_by_type": fig_mortality_by_admission_type(admissions),
        "lab_distributions": fig_lab_distributions(labevents),
        "los_distribution":  fig_los_distribution(icustays),
    }
    print(f"  ✓ {len(figs)} charts generated")

    print("\n[3] Exporting dashboard ...")
    build_dashboard(figs, OUTPUT_FILE)
    print(f"\n✅ Done! Open in browser: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()