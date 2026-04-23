"""
阶段 2-FE：Level 2 特征工程
构建 Track A（死亡率预测）和 Track B（ICU LOS 预测）的特征矩阵。
运行：python stage2_features.py
输出：
  features_track_a.csv  —— Track A 特征矩阵（含标签 hospital_expire_flag）
  features_track_b.csv  —— Track B 特征矩阵（含标签 los）
预期耗时：< 5 min
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = "/Users/pickle/.cache/kagglehub/datasets/asjad99/mimiciii/versions/1/mimic-iii-clinical-database-demo-1.4"


# ════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════

def read(fname, cols=None, dates=None):
    path = os.path.join(DATA_DIR, fname)
    return pd.read_csv(path, usecols=cols, parse_dates=dates, low_memory=False)


def load_chartevents_filtered(itemids):
    """CHARTEVENTS 大文件，分块过滤后合并。"""
    path = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
    usecols = ["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "valuenum", "error"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000,
                              parse_dates=["charttime"], low_memory=False):
        sub = chunk[chunk["itemid"].isin(itemids)].copy()
        # 排除 error 记录
        if "error" in sub.columns:
            sub = sub[sub["error"].isna() | (sub["error"] == 0)]
        if len(sub):
            chunks.append(sub)
    df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"  CHARTEVENTS filtered: {len(df):,} 行")
    return df


# ════════════════════════════════════════════════════════
# 共用特征 1：人口统计学
# ════════════════════════════════════════════════════════

def build_demographics(patients, admissions):
    """
    返回以 hadm_id 为索引的人口统计特征 DataFrame：
      age, age_imputed, gender_M, adm_EMERGENCY, adm_ELECTIVE, adm_URGENT

    age_imputed = 1 表示该患者年龄 >89 岁，dob 被 MIMIC 位移无法精确计算。
    age 填入哨兵值 91，在数值上明确大于 89，保留「高龄」方向信息，
    同时由 age_imputed 标志让模型单独学习该组行为。
    """
    df = admissions[["hadm_id", "subject_id", "admittime",
                      "dischtime", "admission_type", "hospital_expire_flag"]].copy()
    df = df.merge(patients[["subject_id", "dob", "gender"]], on="subject_id", how="left")

    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dob"]       = pd.to_datetime(df["dob"])

    # MIMIC 对 >89 岁患者将 dob 位移至 1800 年代，正常患者 dob >= 1900
    normal_mask = df["dob"] >= pd.Timestamp("1900-01-01")

    df["age"] = np.nan
    df["age_imputed"] = 0  # 0 = 真实年龄，1 = 年龄未知(>89岁，哨兵值 91)

    # 正常患者：精确计算年龄
    df.loc[normal_mask, "age"] = (
        df.loc[normal_mask, "admittime"] - df.loc[normal_mask, "dob"]
    ).dt.days / 365.25

    # >89 岁患者：填哨兵值 91（数值上明确 >89，保留高龄方向信息）+ 打标记
    df.loc[~normal_mask, "age"] = 91.0
    df.loc[~normal_mask, "age_imputed"] = 1

    n_imputed = (~normal_mask).sum()
    print(f"    年龄：正常 {normal_mask.sum()} 条，>89岁哨兵填充 {n_imputed} 条（age=91）")

    # 性别 One-hot
    df["gender_M"] = (df["gender"] == "M").astype(int)

    # 入院类型 One-hot
    for atype in ["EMERGENCY", "ELECTIVE", "URGENT"]:
        df[f"adm_{atype}"] = (df["admission_type"] == atype).astype(int)

    cols = ["hadm_id", "subject_id", "admittime", "dischtime",
            "hospital_expire_flag", "age", "age_imputed", "gender_M",
            "adm_EMERGENCY", "adm_ELECTIVE", "adm_URGENT"]
    return df[cols]


# ════════════════════════════════════════════════════════
# 共用特征 2：Charlson 合并症指数
# ════════════════════════════════════════════════════════

# ICD-9 前缀 → Charlson 疾病分类及权重
# 参考：Quan et al. 2005 更新版
CHARLSON_MAP = {
    # weight: 1
    "MI":          (1, ["410", "412"]),
    "CHF":         (1, ["428"]),
    "PVD":         (1, ["440", "441", "4431", "4432", "4438", "4439", "4471", "5571", "5579", "V434"]),
    "CVD":         (1, ["43", "4379"]),
    "Dementia":    (1, ["290"]),
    "COPD":        (1, ["490", "491", "492", "493", "494", "495", "496", "500", "501", "502", "503", "504", "505"]),
    "Rheumatic":   (1, ["4465", "7100", "7101", "7102", "7103", "7104", "7140", "7141", "7142", "7148"]),
    "PUD":         (1, ["531", "532", "533", "534"]),
    "MildLiver":   (1, ["5712", "5714", "5715", "5716"]),
    "DiabNoComp":  (1, ["2500", "2501", "2502", "2503", "2508", "2509"]),
    # weight: 2
    "DiabComp":    (2, ["2504", "2505", "2506", "2507"]),
    "Hemiplegia":  (2, ["3341", "342", "343", "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449"]),
    "Renal":       (2, ["582", "5830", "5831", "5832", "5834", "5836", "5837", "585", "586", "5880", "V420", "V451", "V56"]),
    "MalignNoMet": (2, ["140", "141", "142", "143", "144", "145", "146", "147", "148", "149",
                        "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",
                        "160", "161", "162", "163", "164", "165",
                        "170", "171", "172", "174", "175", "176",
                        "179", "180", "181", "182", "183", "184", "185", "186", "187", "188", "189",
                        "190", "191", "192", "193", "194", "195",
                        "200", "201", "202", "203", "204", "205", "206", "207", "208",
                        "2386"]),
    "SevereLiver": (3, ["4560", "4561", "4562", "5722", "5723", "5724", "5725", "5726", "5727", "5728"]),
    # weight: 6
    "Metastatic":  (6, ["196", "197", "198", "199"]),
    "AIDS":        (6, ["042", "043", "044"]),
}


def _icd9_match(code, prefixes):
    code = str(code).strip()
    return any(code.startswith(p) for p in prefixes)


def compute_charlson(diagnoses):
    """返回以 hadm_id 为索引的 charlson_index Series。"""
    scores = {}
    for hadm_id, grp in diagnoses.groupby("hadm_id"):
        codes = grp["icd9_code"].dropna().astype(str).tolist()
        total = 0
        for disease, (weight, prefixes) in CHARLSON_MAP.items():
            if any(_icd9_match(c, prefixes) for c in codes):
                total += weight
        scores[hadm_id] = total

    return pd.Series(scores, name="charlson_index").reset_index().rename(
        columns={"index": "hadm_id"}
    )


# ════════════════════════════════════════════════════════
# Track A 专属：入院后 24h Lab 值
# ════════════════════════════════════════════════════════

LAB_ITEMS = {
    50912: "creatinine",
    51006: "bun",
    51301: "wbc",
    51265: "platelets",
    51222: "hemoglobin",
    50885: "bilirubin",
    50813: "lactate",
}

# 合理范围过滤（防止异常值污染统计）
LAB_CLIP = {
    50912: (0, 30),    # creatinine mg/dL
    51006: (0, 300),   # BUN mg/dL
    51301: (0, 200),   # WBC K/uL
    51265: (0, 2000),  # Platelets K/uL
    51222: (0, 25),    # Hemoglobin g/dL
    50885: (0, 50),    # Bilirubin mg/dL
    50813: (0, 30),    # Lactate mmol/L
}


def build_lab_features(labevents, admissions):
    """
    提取每次住院（hadm_id）admittime 后 24h 内的 Lab 值，
    汇总为 min/max/mean，缺失值用全局中位数填充。
    """
    # 只保留目标 itemid
    lab = labevents[labevents["itemid"].isin(LAB_ITEMS)].copy()
    lab["charttime"] = pd.to_datetime(lab["charttime"])

    # 关联 admittime
    adm = admissions[["hadm_id", "admittime"]].copy()
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    lab = lab.merge(adm, on="hadm_id", how="left")

    # 24h 时间窗口过滤
    lab["hours_from_admit"] = (lab["charttime"] - lab["admittime"]).dt.total_seconds() / 3600
    lab = lab[(lab["hours_from_admit"] >= 0) & (lab["hours_from_admit"] <= 24)]

    # 异常值 clip
    for itemid, (lo, hi) in LAB_CLIP.items():
        mask = lab["itemid"] == itemid
        lab.loc[mask, "valuenum"] = lab.loc[mask, "valuenum"].clip(lo, hi)

    # 按 hadm_id + itemid 聚合
    agg = (
        lab.groupby(["hadm_id", "itemid"])["valuenum"]
        .agg(["min", "max", "mean"])
        .reset_index()
    )

    # 宽表 pivot
    rows = []
    for itemid, name in LAB_ITEMS.items():
        sub = agg[agg["itemid"] == itemid][["hadm_id", "min", "max", "mean"]].copy()
        sub = sub.rename(columns={
            "min":  f"{name}_min",
            "max":  f"{name}_max",
            "mean": f"{name}_mean",
        })
        rows.append(sub.set_index("hadm_id"))

    lab_wide = pd.concat(rows, axis=1).reset_index()

    # 缺失值：全局中位数填充
    lab_cols = [c for c in lab_wide.columns if c != "hadm_id"]
    medians = lab_wide[lab_cols].median()
    lab_wide[lab_cols] = lab_wide[lab_cols].fillna(medians)

    print(f"  Lab 特征矩阵：{lab_wide.shape}  ({len(lab_cols)} 个特征列)")
    return lab_wide


# ════════════════════════════════════════════════════════
# Track B 专属：ICU 前 24h 生命体征
# ════════════════════════════════════════════════════════

# 同一体征在 CareVue / MetaVision 下的多个 itemid
VITAL_ITEMS = {
    "hr":   ([211, 220045],   (0, 300)),    # 心率
    "sbp":  ([51, 220050],    (0, 300)),    # 收缩压
    "dbp":  ([8368, 220051],  (0, 200)),    # 舒张压
    "spo2": ([646, 220277],   (50, 100)),   # 血氧
    "temp": ([223761, 678],   (25, 45)),    # 体温（°C / °F 混合，后续统一）
    "rr":   ([618, 220210],   (0, 80)),     # 呼吸率
}

ALL_VITAL_ITEMIDS = [iid for ids, _ in VITAL_ITEMS.values() for iid in ids]


def build_vital_features(chartevents, icustays):
    """
    提取每次 ICU 入住（icustay_id）intime 后 24h 内的生命体征，
    汇总为 min/max/mean，缺失值用全局中位数填充。
    """
    ce = chartevents[chartevents["itemid"].isin(ALL_VITAL_ITEMIDS)].copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"])

    icu = icustays[["icustay_id", "hadm_id", "intime"]].copy()
    icu["intime"] = pd.to_datetime(icu["intime"])
    ce = ce.merge(icu, on="icustay_id", how="left")

    # 24h 时间窗口
    ce["hours_from_intime"] = (ce["charttime"] - ce["intime"]).dt.total_seconds() / 3600
    ce = ce[(ce["hours_from_intime"] >= 0) & (ce["hours_from_intime"] <= 24)]

    rows = []
    for name, (itemids, (lo, hi)) in VITAL_ITEMS.items():
        sub = ce[ce["itemid"].isin(itemids)].copy()
        sub["valuenum"] = sub["valuenum"].clip(lo, hi)

        # 体温：CareVue(678) 使用华氏度，转摄氏度
        if name == "temp":
            mask_f = sub["itemid"] == 678
            sub.loc[mask_f, "valuenum"] = (sub.loc[mask_f, "valuenum"] - 32) * 5 / 9
            sub["valuenum"] = sub["valuenum"].clip(25, 45)  # 再 clip 一次

        agg = (
            sub.groupby("icustay_id")["valuenum"]
            .agg(**{
                f"{name}_min":  "min",
                f"{name}_max":  "max",
                f"{name}_mean": "mean",
            })
            .reset_index()
        )
        rows.append(agg.set_index("icustay_id"))

    vital_wide = pd.concat(rows, axis=1).reset_index()

    # 关联 hadm_id
    vital_wide = vital_wide.merge(icu[["icustay_id", "hadm_id"]], on="icustay_id", how="left")

    # 缺失值：全局中位数填充
    vital_cols = [c for c in vital_wide.columns if c not in ["icustay_id", "hadm_id"]]
    medians = vital_wide[vital_cols].median()
    vital_wide[vital_cols] = vital_wide[vital_cols].fillna(medians)

    print(f"  体征特征矩阵：{vital_wide.shape}  ({len(vital_cols)} 个特征列)")
    return vital_wide


# ════════════════════════════════════════════════════════
# Track B 专属：诊断特征
# ════════════════════════════════════════════════════════

def build_diagnosis_features(diagnoses):
    """每次住院的诊断数量。"""
    diag_count = (
        diagnoses.groupby("hadm_id")["icd9_code"]
        .count()
        .reset_index()
        .rename(columns={"icd9_code": "diag_count"})
    )
    return diag_count


# ════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("阶段 2-FE：特征工程")
    print("=" * 55)

    # ── 加载原始表 ──────────────────────────────────────
    print("\n[1] 加载原始数据 ...")
    patients   = read("PATIENTS.csv",
                      cols=["subject_id", "gender", "dob"],
                      dates=["dob"])
    admissions = read("ADMISSIONS.csv",
                      cols=["subject_id", "hadm_id", "admittime", "dischtime",
                            "admission_type", "hospital_expire_flag"],
                      dates=["admittime", "dischtime"])
    icustays   = read("ICUSTAYS.csv",
                      cols=["subject_id", "hadm_id", "icustay_id",
                            "first_careunit", "intime", "outtime", "los"],
                      dates=["intime", "outtime"])
    diagnoses  = read("DIAGNOSES_ICD.csv",
                      cols=["subject_id", "hadm_id", "icd9_code"])
    labevents  = read("LABEVENTS.csv",
                      cols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
                      dates=["charttime"])

    admissions["hospital_expire_flag"] = admissions["hospital_expire_flag"].fillna(0).astype(int)

    print(f"  PATIENTS:    {len(patients):,}")
    print(f"  ADMISSIONS:  {len(admissions):,}")
    print(f"  ICUSTAYS:    {len(icustays):,}")
    print(f"  DIAGNOSES:   {len(diagnoses):,}")
    print(f"  LABEVENTS:   {len(labevents):,}")

    # ── 共用特征 ────────────────────────────────────────
    print("\n[2] 构建人口统计学特征 ...")
    demo = build_demographics(patients, admissions)
    print(f"  人口统计特征：{demo.shape}")

    print("\n[3] 计算 Charlson 合并症指数 ...")
    charlson = compute_charlson(diagnoses)
    print(f"  Charlson 指数：{charlson.shape}  均值={charlson['charlson_index'].mean():.2f}")

    # ── Track A ─────────────────────────────────────────
    print("\n[4] 构建 Track A Lab 特征（入院后 24h）...")
    lab_feats = build_lab_features(labevents, admissions)

    print("\n[5] 组装 Track A 特征矩阵 ...")
    track_a = (
        demo
        .merge(charlson, on="hadm_id", how="left")
        .merge(lab_feats,  on="hadm_id", how="left")
    )
    # Charlson 缺失 → 0（该次住院无诊断记录）
    track_a["charlson_index"] = track_a["charlson_index"].fillna(0)

    # 目标列
    target_a = "hospital_expire_flag"
    feature_cols_a = [c for c in track_a.columns
                      if c not in ["hadm_id", "subject_id", "admittime",
                                   "dischtime", target_a]]
    print(f"  Track A 特征数：{len(feature_cols_a)}  样本数：{len(track_a)}")
    print(f"  死亡率：{track_a[target_a].mean()*100:.1f}%")
    track_a.to_csv("features_track_a.csv", index=False)
    print("  ✓ 已保存 features_track_a.csv")

    # ── Track B ─────────────────────────────────────────
    print("\n[6] 加载并过滤 CHARTEVENTS（体征 itemid）...")
    chartevents = load_chartevents_filtered(ALL_VITAL_ITEMIDS)

    print("\n[7] 构建 Track B 体征特征（ICU 前 24h）...")
    vital_feats = build_vital_features(chartevents, icustays)

    print("\n[8] 构建诊断数量特征 ...")
    diag_feats = build_diagnosis_features(diagnoses)

    print("\n[9] 组装 Track B 特征矩阵 ...")
    # Track B 以 icustay 为粒度
    icu_base = icustays[["icustay_id", "hadm_id", "subject_id", "los"]].copy()

    track_b = (
        icu_base
        .merge(demo.drop(columns=["subject_id", "dischtime", "hospital_expire_flag"]),
               on="hadm_id", how="left")
        .merge(charlson,    on="hadm_id", how="left")
        .merge(diag_feats,  on="hadm_id", how="left")
        .merge(vital_feats.drop(columns=["hadm_id"]), on="icustay_id", how="left")
    )
    track_b["charlson_index"] = track_b["charlson_index"].fillna(0)
    track_b["diag_count"]     = track_b["diag_count"].fillna(0)

    target_b = "los"
    feature_cols_b = [c for c in track_b.columns
                      if c not in ["icustay_id", "hadm_id", "subject_id",
                                   "admittime", target_b]]
    print(f"  Track B 特征数：{len(feature_cols_b)}  样本数：{len(track_b)}")
    print(f"  LOS 中位数：{track_b[target_b].median():.2f} 天")
    track_b.to_csv("features_track_b.csv", index=False)
    print("  ✓ 已保存 features_track_b.csv")

    print("\n✅ 阶段 2-FE 完成！")
    print(f"   Track A：features_track_a.csv  ({len(track_a)} 行 × {len(feature_cols_a)+1} 列含标签)")
    print(f"   Track B：features_track_b.csv  ({len(track_b)} 行 × {len(feature_cols_b)+1} 列含标签)")

    return track_a, track_b, feature_cols_a, feature_cols_b


if __name__ == "__main__":
    main()