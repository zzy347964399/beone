"""
阶段 0：数据加载与验证
运行：python stage0_load.py
预期耗时：< 2 min
"""

import pandas as pd
import os

DATA_DIR = "/Users/pickle/.cache/kagglehub/datasets/asjad99/mimiciii/versions/1/mimic-iii-clinical-database-demo-1.4"

# ── 核心表清单 ──────────────────────────────────────────
CORE_TABLES = {
    "PATIENTS":       ("PATIENTS.csv",       ["subject_id", "gender", "dob", "dod"]),
    "ADMISSIONS":     ("ADMISSIONS.csv",     ["subject_id", "hadm_id", "admittime", "dischtime",
                                               "admission_type", "hospital_expire_flag"]),
    "ICUSTAYS":       ("ICUSTAYS.csv",       ["subject_id", "hadm_id", "icustay_id",
                                               "first_careunit", "intime", "outtime", "los"]),
    "DIAGNOSES_ICD":  ("DIAGNOSES_ICD.csv",  ["subject_id", "hadm_id", "icd9_code"]),
    "D_ICD_DIAGNOSES":("D_ICD_DIAGNOSES.csv",["icd9_code", "long_title"]),
    "LABEVENTS":      ("LABEVENTS.csv",      ["subject_id", "hadm_id", "itemid", "charttime", "valuenum"]),
    "CHARTEVENTS":    ("CHARTEVENTS.csv",    ["subject_id", "hadm_id", "icustay_id",
                                               "itemid", "charttime", "valuenum", "error"]),
    "PRESCRIPTIONS":  ("PRESCRIPTIONS.csv",  ["subject_id", "hadm_id", "drug"]),
}

# Lab 指标 itemid（用于 LABEVENTS 快速过滤）
LAB_ITEMIDS = [50861, 50885, 50912, 50813, 51006, 51222, 51265, 51301]

# 体征 itemid（用于 CHARTEVENTS 快速过滤）
VITAL_ITEMIDS = [211, 220045, 51, 220050, 8368, 220051, 646, 220277, 223761, 678, 618, 220210]


def load_table(name, filename, usecols=None, parse_dates=None):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"  ✗ 文件不存在：{path}")
        return None
    # 只加载需要的列（节省内存）
    df = pd.read_csv(path, usecols=usecols, parse_dates=parse_dates, low_memory=False)
    print(f"  ✓ {name:<20} {len(df):>8,} 行  列：{list(df.columns)}")
    return df


def load_chartevents_filtered(itemids):
    """CHARTEVENTS 体积大，按 itemid 分块过滤后合并，避免 OOM。"""
    path = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
    if not os.path.exists(path):
        print(f"  ✗ 文件不存在：{path}")
        return None

    usecols = ["subject_id", "hadm_id", "icustay_id", "itemid", "charttime", "valuenum", "error"]
    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000,
                              parse_dates=["charttime"], low_memory=False):
        filtered = chunk[chunk["itemid"].isin(itemids)]
        if len(filtered):
            chunks.append(filtered)

    if not chunks:
        print("  ✗ CHARTEVENTS：过滤后无数据")
        return None

    df = pd.concat(chunks, ignore_index=True)
    # 排除错误记录
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == 0)]
    print(f"  ✓ CHARTEVENTS (filtered)    {len(df):>8,} 行  (itemids={itemids})")
    return df


def validate_admissions(df):
    """核查 ADMISSIONS 关键字段。"""
    print("\n── ADMISSIONS 健康检查 ──")
    n_total    = len(df)
    n_null_flag = df["hospital_expire_flag"].isna().sum()
    # 陷阱：hospital_expire_flag 有 NULL，必须 fillna(0) 后再 sum，否则死亡率偏低
    flag_clean = df["hospital_expire_flag"].fillna(0).astype(int)
    n_expired  = flag_clean.sum()
    print(f"  总住院次数：{n_total}")
    print(f"  hospital_expire_flag NULL：{n_null_flag}  (已填充为 0)")
    print(f"  院内死亡：{n_expired}  ({n_expired/n_total*100:.1f}%)")
    print(f"  入院类型分布：\n{df['admission_type'].value_counts().to_string()}")


def validate_icustays(df):
    """核查 ICUSTAYS。"""
    print("\n── ICUSTAYS 健康检查 ──")
    print(f"  ICU 住院次数：{len(df)}")
    print(f"  LOS（天）中位数：{df['los'].median():.2f}")
    print(f"  护理单元分布：\n{df['first_careunit'].value_counts().to_string()}")


def main():
    print("=" * 55)
    print("阶段 0：数据加载验证")
    print("=" * 55)

    dfs = {}

    # ── 加载小表 ──
    print("\n[1] 加载核心小表 ...")
    for name, (fname, cols) in CORE_TABLES.items():
        if name == "CHARTEVENTS":
            continue  # 大表单独处理
        parse_dates = None
        if name == "PATIENTS":
            parse_dates = ["dob", "dod"]
        elif name == "ADMISSIONS":
            parse_dates = ["admittime", "dischtime"]
        elif name == "ICUSTAYS":
            parse_dates = ["intime", "outtime"]
        elif name == "LABEVENTS":
            parse_dates = ["charttime"]

        dfs[name] = load_table(name, fname, usecols=cols, parse_dates=parse_dates)

    # ── 加载 LABEVENTS（中等大小，全量加载可接受）──
    print("\n[2] 加载 LABEVENTS（全量）...")
    lab_path = os.path.join(DATA_DIR, "LABEVENTS.csv")
    if os.path.exists(lab_path):
        dfs["LABEVENTS"] = pd.read_csv(
            lab_path,
            usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"],
            parse_dates=["charttime"],
            low_memory=False,
        )
        print(f"  ✓ LABEVENTS              {len(dfs['LABEVENTS']):>8,} 行")

    # ── 加载 CHARTEVENTS（分块过滤）──
    print("\n[3] 加载 CHARTEVENTS（分块过滤，仅保留体征 itemid）...")
    dfs["CHARTEVENTS"] = load_chartevents_filtered(VITAL_ITEMIDS)

    # ── 健康检查 ──
    print("\n[4] 核心表健康检查 ...")
    if dfs.get("ADMISSIONS") is not None:
        validate_admissions(dfs["ADMISSIONS"])
    if dfs.get("ICUSTAYS") is not None:
        validate_icustays(dfs["ICUSTAYS"])

    print("\n✅ 阶段 0 完成，数据已就绪，继续执行 stage1a_eda.py")
    return dfs


if __name__ == "__main__":
    main()