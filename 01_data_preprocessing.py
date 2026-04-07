import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
import joblib
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 日志：同时输出到控制台 + 文件
# ============================================================
class Logger:
    def __init__(self, filename="preprocessing_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

sys.stdout = Logger("preprocessing_log.txt")

print("=" * 70)
print("数据预处理升级版 v2 — Student Dropout Dataset")
print("改动：列名清洗 / 特征工程 / 去冗余 / Split后Scaler / 稀疏列处理")
print("=" * 70)

# ============================================================
# STEP 1  读取数据 & 列名规范化
# ============================================================
print("\n[STEP 1] 读取数据 & 列名清洗")

df = pd.read_csv("data.csv", sep=";")
print(f"  原始数据形状: {df.shape}")

# 去除 BOM头、Tab、首尾空格
df.columns = [c.strip().replace('\t', '').replace('\ufeff', '') for c in df.columns]
print(f"  列名清洗完成，共 {len(df.columns)} 列")
print(f"  缺失值总数: {df.isnull().sum().sum()}")  # 官方已预处理，应为 0

# ============================================================
# STEP 2  特征工程 — 新增派生特征
# ============================================================
print("\n[STEP 2] 特征工程：构建派生特征")

# --- 2.1  年龄生命周期分组（替代连续 Age，消除离群点杠杆效应）---
def age_to_lifecycle(age):
    if age <= 21:   return 0  # traditional：应届生
    elif age <= 30: return 1  # young_adult：青年
    elif age <= 40: return 2  # adult：成年
    else:           return 3  # mature：大龄学生

df['age_group'] = df['Age at enrollment'].apply(age_to_lifecycle)
age_counts = df['age_group'].value_counts().sort_index()
print(f"  age_group 分布: {dict(age_counts)}"
      f"  (0=≤21岁, 1=22-30, 2=31-40, 3=>40)")

# --- 2.2  未参评标志：grade=0 且 enrolled>0 → 实为缺考，非真实0分 ---
df['no_eval_sem1'] = (
    (df['Curricular units 1st sem (enrolled)'] > 0) &
    (df['Curricular units 1st sem (grade)'] == 0)
).astype(int)

df['no_eval_sem2'] = (
    (df['Curricular units 2nd sem (enrolled)'] > 0) &
    (df['Curricular units 2nd sem (grade)'] == 0)
).astype(int)

print(f"  no_eval_sem1=1: {df['no_eval_sem1'].sum()} 条"
      f"  ({df['no_eval_sem1'].mean()*100:.1f}%)")
print(f"  no_eval_sem2=1: {df['no_eval_sem2'].sum()} 条"
      f"  ({df['no_eval_sem2'].mean()*100:.1f}%)")

# --- 2.3  课程通过率（消除注册门数差异） ---
df['pass_rate_sem1'] = (
    df['Curricular units 1st sem (approved)'] /
    df['Curricular units 1st sem (enrolled)'].replace(0, np.nan)
).fillna(0).clip(0, 1)

df['pass_rate_sem2'] = (
    df['Curricular units 2nd sem (approved)'] /
    df['Curricular units 2nd sem (enrolled)'].replace(0, np.nan)
).fillna(0).clip(0, 1)

print(f"  pass_rate_sem1 均值: {df['pass_rate_sem1'].mean():.3f}")
print(f"  pass_rate_sem2 均值: {df['pass_rate_sem2'].mean():.3f}")

# --- 2.4  成绩变化趋势（正=进步，负=退步；两学期均有成绩才计算） ---
both_graded = (
    (df['Curricular units 1st sem (grade)'] > 0) &
    (df['Curricular units 2nd sem (grade)'] > 0)
)
df['grade_trend'] = np.where(
    both_graded,
    df['Curricular units 2nd sem (grade)'] -
    df['Curricular units 1st sem (grade)'],
    0
)
print(f"  grade_trend 有效计算行数: {both_graded.sum()} 条")

# --- 2.5  经济压力综合标志（欠费 OR 有债务 → 经济困难） ---
df['economic_stress'] = (
    (df['Debtor'] == 1) | (df['Tuition fees up to date'] == 0)
).astype(int)
print(f"  economic_stress=1: {df['economic_stress'].sum()} 条"
      f"  ({df['economic_stress'].mean()*100:.1f}%)")

# --- 2.6  家庭教育资本（父母学历均值，在分组映射后计算，见 STEP 3） ---
# 先保留原始列，分组后在 STEP 3 末尾计算

print("  ✅ 特征工程完成，新增 7 个派生特征")

# ============================================================
# STEP 3  分组映射函数（高基数类别）
# ============================================================
print("\n[STEP 3] 高基数变量分组")

# --- 3.1  国籍：21 种 → 5 组 ---
def group_nationality(x):
    if x == 1:                              return 0  # 葡萄牙
    elif x in [2,6,11,13,14,17,32,62,
               100,103,105]:               return 1  # 欧洲
    elif x in [41,101,108,109]:            return 2  # 美洲
    elif x in [21,22,24,25,26]:            return 3  # 非洲葡语国家
    else:                                  return 4  # 其他/亚洲

# --- 3.2  父母学历：29-34 种 → 5 组（0=文盲/未知 … 4=高等教育）---
def group_qualification(x):
    if   x in [34, 35, 36]:               return 0  # 文盲/不识字/未知
    elif x in [14,26,29,30,37,38]:        return 1  # 基础教育（小/初中）
    elif x in [1,9,10,11,12,18,19,22,27]: return 2  # 中等教育/高中
    elif x in [39, 42]:                   return 3  # 技术/专业/专科
    elif x in [2,3,4,5,6,40,41,43,44]:   return 4  # 高等教育（本/硕/博）
    else:                                  return 0  # 兜底→未知

# --- 3.3  父母职业：32-46 种 → 5 组 ---
def group_occupation(x):
    if   x in [0, 90, 99]:                        return 0  # 无业/学生/未知
    elif x in [1, 10]:                             return 1  # 管理/高级官员
    elif x in [2,122,123,125]:                     return 2  # 专业人员
    elif x in [3,131,132,134,
               4,141,143,144,
               5,151,152,153]:                     return 3  # 中级技术/行政/服务
    elif x in [6,7,8,9,
               171,173,175,
               191,192,193,194]:                   return 4  # 蓝领/技工/农业
    else:                                          return 0  # 兜底

# --- 3.4  Previous qualification：17 种编码 → 有序 5 级 ---
# 原始编码并非按学历升序，手动重映射为 0-4 有序整数
prev_qual_map = {
    # 0：基础教育（初中及以下）
    19: 0, 38: 0, 14: 0, 15: 0,
    # 1：高中（未完成或完成）
    1: 1, 9: 1, 10: 1, 12: 1,
    # 2：技术/专科
    39: 2, 42: 2,
    # 3：大专/本科在读/本科
    6: 3, 40: 3,
    # 4：本科及以上（学位/硕士/博士）
    2: 4, 3: 4, 4: 4, 5: 4, 43: 4,
}
df['prev_qual_ordinal'] = df['Previous qualification'].map(prev_qual_map).fillna(1)

# 执行所有分组
df['Nacionality_grouped']  = df['Nacionality'].apply(group_nationality)
df['Mother_qual_grouped']  = df["Mother's qualification"].apply(group_qualification)
df['Father_qual_grouped']  = df["Father's qualification"].apply(group_qualification)
df['Mother_occ_grouped']   = df["Mother's occupation"].apply(group_occupation)
df['Father_occ_grouped']   = df["Father's occupation"].apply(group_occupation)

# 2.6（续）家庭教育资本：分组完成后计算
df['family_edu_capital'] = (df['Mother_qual_grouped'] + df['Father_qual_grouped']) / 2

print(f"  Nacionality_grouped 分布: {dict(df['Nacionality_grouped'].value_counts().sort_index())}")
print(f"  Mother_qual_grouped 分布: {dict(df['Mother_qual_grouped'].value_counts().sort_index())}")
print(f"  Father_qual_grouped 分布: {dict(df['Father_qual_grouped'].value_counts().sort_index())}")
print(f"  prev_qual_ordinal 分布:   {dict(df['prev_qual_ordinal'].value_counts().sort_index())}")
print("  ✅ 分组映射完成")

# ============================================================
# STEP 4  去冗余列
# ============================================================
print("\n[STEP 4] 去除冗余/极端稀疏特征")

# International 与 Nacionality 高度重叠（97.5% 为葡萄牙人，International≈非葡）
corr_intl = df['International'].corr(
    (df['Nacionality'] != 1).astype(int))
print(f"  International 与 非葡萄牙标志 相关系数: {corr_intl:.3f} → 删除 International")
df.drop(columns=['International'], inplace=True)

# 删除原始 Previous qualification（已映射为 prev_qual_ordinal，保留有序版）
# 删除原始 Age at enrollment（已生成 age_group，保留分组版；连续版留着用于标准化）
# 注：两者都保留，让下游选择用哪个；原始 Nacionality 留到独热后再删

# 删除极端稀疏的 Application mode 子类（正例率 < 0.5%，在 one-hot 之前合并）
# Application mode 2/10/26/27/57 各自 ≤10 行，合并为"其他渠道"类别
rare_app_modes = [2, 10, 26, 27, 57]
df['Application mode'] = df['Application mode'].apply(
    lambda x: 99 if x in rare_app_modes else x
)
sparse_merged = df['Application mode'].value_counts()[99] if 99 in df['Application mode'].values else 0
print(f"  Application mode 稀疏类别 (2/10/26/27/57) 合并为 99，共 {sparse_merged} 条")

print("  ✅ 冗余列处理完成")

# ============================================================
# STEP 5  目标变量编码
# ============================================================
print("\n[STEP 5] 目标变量编码")

le = LabelEncoder()
df['Target_encoded'] = le.fit_transform(df['Target'])
label_mapping = {k: int(v) for k, v in zip(le.classes_, le.transform(le.classes_))}
print(f"  标签映射: {label_mapping}")
print(f"  类别分布: {dict(df['Target_encoded'].value_counts().sort_index())}")
# 额外保存二分类版本（Dropout=1 vs 非Dropout=0）
df['Target_binary'] = (df['Target'] == 'Dropout').astype(int)
print(f"  二分类 (Dropout=1): {df['Target_binary'].sum()} 条")
df.drop(columns=['Target'], inplace=True)

# ============================================================
# STEP 6  独热编码（仅低基数名义类别）
# ============================================================
print("\n[STEP 6] 独热编码")

# 要独热的列（低基数名义类别 + 分组后的高基数）
one_hot_cols = [
    'Marital status',         # 6 类（单身/已婚/…）
    'Application mode',       # 合并后 14 类
    'Course',                 # 17 类
    'Daytime/evening attendance',  # 2类（白天/晚上）
    'Nacionality_grouped',    # 5 组
    'Mother_qual_grouped',    # 5 组
    'Father_qual_grouped',    # 5 组
    'Mother_occ_grouped',     # 5 组
    'Father_occ_grouped',     # 5 组
    'age_group',              # 4 组（新增派生）
]

df_encoded = pd.get_dummies(df, columns=one_hot_cols, drop_first=True, dtype=int)
print(f"  独热编码后形状: {df_encoded.shape}")

# 删除已替代的原始高基数列
drop_orig = [
    'Nacionality',
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    'Previous qualification',   # 已由 prev_qual_ordinal 替代
    'Age at enrollment',        # 已由 age_group one-hot 替代（连续版已用于特征工程）
]
# 只删除实际存在的列（防御性处理）
drop_orig = [c for c in drop_orig if c in df_encoded.columns]
df_encoded.drop(columns=drop_orig, inplace=True)
print(f"  删除原始高基数列后形状: {df_encoded.shape}")
print("  ✅ 独热编码完成")

# ============================================================
# STEP 7  分离 X / y，然后 Train/Val/Test 分割
#         ！！scaler 必须在 split 之后只在训练集 fit！！
# ============================================================
print("\n[STEP 7] 特征/目标分离 & 数据集分割（stratified 70-10-20）")

TARGET_COL      = 'Target_encoded'
TARGET_BIN_COL  = 'Target_binary'

X = df_encoded.drop(columns=[TARGET_COL, TARGET_BIN_COL])
y = df_encoded[TARGET_COL]
y_bin = df_encoded[TARGET_BIN_COL]

print(f"  X 形状: {X.shape}   y 形状: {y.shape}")

# 第一刀：70% 训练，30% 临时
X_train, X_temp, y_train, y_temp, y_bin_train, y_bin_temp = train_test_split(
    X, y, y_bin,
    test_size=0.30, random_state=42, stratify=y
)

# 第二刀：30% 中拆出 10% 验证(1/3) + 20% 测试(2/3)
X_val, X_test, y_val, y_test, y_bin_val, y_bin_test = train_test_split(
    X_temp, y_temp, y_bin_temp,
    test_size=0.6667, random_state=42, stratify=y_temp
)

print(f"  训练集: {X_train.shape}  "
      f"验证集: {X_val.shape}  "
      f"测试集: {X_test.shape}")

# 各集合类别分布验证
for name, yy in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    dist = yy.value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"  {name} Target分布: {dist}")

# ============================================================
# STEP 8  标准化（仅连续/有序列；在训练集 fit，其余 transform）
# ============================================================
print("\n[STEP 8] StandardScaler（训练集 fit，测试集/验证集仅 transform）")

# 确定需要标准化的列：连续数值型（nunique > 10）
# 独热列（0/1）不做标准化，避免极端 z-score 问题
binary_like = [c for c in X_train.columns if X_train[c].nunique() <= 2]
scale_cols  = [c for c in X_train.columns if c not in binary_like]

print(f"  标准化列数: {len(scale_cols)}")
print(f"  保持原值列数（二值/独热）: {len(binary_like)}")

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_val_scaled   = X_val.copy()
X_test_scaled  = X_test.copy()

# fit 只在训练集
X_train_scaled[scale_cols] = scaler.fit_transform(X_train[scale_cols])
# val/test 只 transform
X_val_scaled[scale_cols]   = scaler.transform(X_val[scale_cols])
X_test_scaled[scale_cols]  = scaler.transform(X_test[scale_cols])

print(f"  训练集连续列均值范围: "
      f"{X_train_scaled[scale_cols].mean().min():.3f} ~ "
      f"{X_train_scaled[scale_cols].mean().max():.3f}  (应接近 0)")
print(f"  测试集连续列均值范围: "
      f"{X_test_scaled[scale_cols].mean().min():.3f} ~ "
      f"{X_test_scaled[scale_cols].mean().max():.3f}  (允许略偏离 0)")
print("  ✅ 标准化完成（无数据泄漏）")

# ============================================================
# STEP 9  构建不含 StandardScaler 的全量数据集（供 t-SNE 使用）
# ============================================================
print("\n[STEP 9] 构建完整数据集（供 t-SNE / 聚类分析使用）")

# 对全量 X 做标准化（t-SNE 是纯可视化，不涉及模型评估，轻微泄漏可接受；
# 仍使用训练集 fit 的 scaler 保持一致性）
X_full_scaled = X.copy()
X_full_scaled[scale_cols] = scaler.transform(X[scale_cols])

full_df = X_full_scaled.copy()
full_df['Target']        = y.values
full_df['Target_binary'] = y_bin.values
print(f"  完整预处理数据形状: {full_df.shape}")

# ============================================================
# STEP 10  保存所有输出文件
# ============================================================
print("\n[STEP 10] 保存文件")

# 10.1 分割数据集（带 Target）
train_out = X_train_scaled.copy()
train_out['Target']        = y_train.values
train_out['Target_binary'] = y_bin_train.values
train_out.to_csv("train_70.csv", index=False)

val_out = X_val_scaled.copy()
val_out['Target']        = y_val.values
val_out['Target_binary'] = y_bin_val.values
val_out.to_csv("val_10.csv", index=False)

test_out = X_test_scaled.copy()
test_out['Target']        = y_test.values
test_out['Target_binary'] = y_bin_test.values
test_out.to_csv("test_20.csv", index=False)

# 10.2 完整数据（供 t-SNE）
full_df.to_csv("full_preprocessed_data.csv", index=False)

# 10.3 辅助文件
with open("label_mapping.json", "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, ensure_ascii=False, indent=2)

joblib.dump(scaler, "standard_scaler.pkl")

with open("feature_names.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(X_train_scaled.columns))

with open("scale_cols.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(scale_cols))

print("  📁 train_70.csv       — 训练集（含 Target + Target_binary）")
print("  📁 val_10.csv         — 验证集")
print("  📁 test_20.csv        — 测试集")
print("  📁 full_preprocessed_data.csv — 全量数据（供 t-SNE/聚类）")
print("  📁 label_mapping.json — 标签映射")
print("  📁 standard_scaler.pkl— Scaler 对象（可复现）")
print("  📁 feature_names.txt  — 特征名列表")
print("  📁 scale_cols.txt     — 标准化列名列表")

# ============================================================
# STEP 11  预处理摘要报告
# ============================================================
print("\n" + "=" * 70)
print("预处理完成摘要")
print("=" * 70)
print(f"  原始数据:      4424 行 × 37 列")
print(f"  最终特征数:    {X_train_scaled.shape[1]} 列")
print(f"  新增派生特征:  7 个（age_group / no_eval×2 / pass_rate×2 / grade_trend / economic_stress / family_edu_capital）")
print(f"  删除冗余列:    International（与国籍重叠）/ 原始 Previous qualification / 原始 Age")
print(f"  合并稀疏类:    Application mode 中 5 个极稀疏类别合并为 '99-其他渠道'")
print(f"  独热编码:      仅对名义类别列执行，二值/连续列保持原值")
print(f"  标准化策略:    仅连续/有序列，训练集 fit → 验证/测试集 transform（无泄漏）")
print(f"  目标变量:      三分类（0=Dropout, 1=Enrolled, 2=Graduate）+ 二分类版本")
print(f"  数据分割:      stratify=y，70% 训练 / 10% 验证 / 20% 测试")
print(f"  类别不均衡:    建议建模时加 class_weight='balanced'")
print("=" * 70)
print("✅ 全部完成，可直接用于 t-SNE / 聚类 / 监督学习任务")
