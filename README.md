# Task 1 — Data Preprocessing

**Dataset:** Predict Students' Dropout and Academic Success  
**Source:** UCI Machine Learning Repository (ID: 697)  
**Script:** `data_preprocessing.py`  
**Output:** `full_preprocessed_data.csv` + split files + auxiliary files

---

## 1. 数据集概览

| 项目 | 数值 |
|------|------|
| 原始样本数 | 4,424 |
| 原始特征数 | 36 |
| 目标变量类别 | 3（Graduate / Dropout / Enrolled）|
| 缺失值 | 0（官方已预处理）|
| 处理后特征数 | **89** |
| 处理后文件大小 | 4,424 行 × 91 列（含 Target + Target_binary）|

**目标变量分布（类别不均衡）：**

| 类别 | 编码 | 样本数 | 占比 |
|------|------|--------|------|
| Graduate | 2 | 2,209 | 49.9% |
| Dropout  | 0 | 1,421 | 32.1% |
| Enrolled | 1 | 794   | 18.0% |

---

## 2. 处理流程总览

```
原始 data.csv (4424×37)
    │
    ├─ Step 1  列名规范化（去除 BOM / Tab 字符）
    ├─ Step 2  特征工程（新增 8 个派生特征）
    ├─ Step 3  高基数变量分组映射
    ├─ Step 4  去除冗余 & 合并极稀疏类别
    ├─ Step 5  目标变量编码（三分类 + 二分类）
    ├─ Step 6  独热编码（仅名义类别列）
    ├─ Step 7  Train / Val / Test 分割（70 / 10 / 20）
    ├─ Step 8  StandardScaler（仅连续列，训练集 fit）
    └─ Step 9  保存所有输出文件

输出：full_preprocessed_data.csv (4424×91)
      train_70.csv / val_10.csv / test_20.csv
```

---

## 3. 各步骤详细说明

### Step 1 — 列名规范化

原始 CSV 文件存在两个列名污染问题：

- 文件开头含有 **BOM 字符**（`\ufeff`），导致第一列名读取异常
- `Daytime/evening attendance` 列名末尾含有**制表符**（`\t`），独热编码后会污染新列名

处理方式：

```python
df.columns = [c.strip().replace('\t', '').replace('\ufeff', '') for c in df.columns]
```

---

### Step 2 — 特征工程

在原始 36 个特征基础上，新增 **8 个派生特征**，提升数据的信息密度和模型可解释性。

#### 2.1 年龄生命周期分组 `age_group`

原始 `Age at enrollment` 是连续变量，分布极度右偏（中位数 20 岁，最大值 70 岁），IQR 法检测出 441 个"离群点"（> 34 岁）。这些不是数据错误，而是真实的成人学生（在职继续教育）。

处理策略：**不删除，转化为生命周期阶段**，消除极端值对基于距离算法的杠杆效应。

| 编码 | 区间 | 含义 | 样本数 |
|------|------|------|--------|
| 0 | ≤ 21 岁 | 应届生（参照基准，drop_first 后删除）| 2,873 |
| 1 | 22–30 岁 | 青年学生 | 889 |
| 2 | 31–40 岁 | 成年学生 | 437 |
| 3 | > 40 岁  | 大龄学生 | 225 |

#### 2.2 未参评标志 `no_eval_sem1` / `no_eval_sem2`

数据集中第一学期成绩（grade）为 0 的记录有 **718 条（16.2%）**，第二学期有 **870 条（19.7%）**。这些 0 **并非真实的零分**，而是该学生注册了课程但未参加任何考试（evaluations = 0）——即"缺考"。

若直接用 0 参与成绩均值计算，会严重拉低该生的平均分，误导模型。

处理策略：**构建布尔标志变量，保留原始 0 值**。

```
no_eval_sem1 = 1  当 enrolled > 0 且 grade == 0
```

| 特征 | 正例数 | 正例率 |
|------|--------|--------|
| `no_eval_sem1` | 538 | 12.2% |
| `no_eval_sem2` | 690 | 15.6% |

经验证：第一学期 approved = 0 的 718 条记录中，**79.4% 最终 Dropout**，这是数据集中最强的早期退学信号之一。

#### 2.3 课程通过率 `pass_rate_sem1` / `pass_rate_sem2`

原始 `approved`（通过课程数）的绝对值受注册门数影响：注册 2 门通过 2 门，与注册 8 门通过 2 门，信号完全不同。

处理策略：**构建相对通过率**，消除注册门数差异。

```
pass_rate_sem1 = approved_1st / enrolled_1st
                （enrolled = 0 时填 0，clip 到 [0, 1]）
```

| 特征 | 均值（标准化前）|
|------|----------------|
| `pass_rate_sem1` | 0.698 |
| `pass_rate_sem2` | 0.660 |

#### 2.4 成绩变化趋势 `grade_trend`

捕捉学生在两个学期之间的学业进步或退步趋势，对识别"开始好后来差"的潜在退学风险有价值。

```
grade_trend = grade_2nd - grade_1st
             （仅对两学期均有成绩的 3,512 条计算，其余填 0）
```

正值表示进步，负值表示退步。

#### 2.5 经济压力综合标志 `economic_stress`

`Debtor`（有债务）和 `Tuition fees up to date`（学费未按时缴纳）均反映经济困难，两者相关系数为 -0.408。将其合并为一个综合经济压力指标：

```
economic_stress = 1  当 Debtor == 1 OR Tuition fees up to date == 0
```

正例数：785 条（17.7%）

#### 2.6 前置学历有序映射 `prev_qual_ordinal`

原始 `Previous qualification` 有 17 种编码，但编码值顺序并非按学历升序排列（如编码 19 = 初中，编码 6 = 在读高等教育）。旧版直接作为数值使用，存在语义错误。

新版按实际教育层次**手动重映射为 0–4 有序整数**：

| 有序值 | 含义 | 原始编码 |
|--------|------|----------|
| 0 | 基础教育（初中及以下）| 14, 15, 19, 38 |
| 1 | 高中（完成或未完成）  | 1, 9, 10, 12（最多，3,777 条）|
| 2 | 技术 / 专科          | 39, 42 |
| 3 | 大专 / 本科在读       | 6, 40 |
| 4 | 本科及以上           | 2, 3, 4, 5, 43 |

#### 2.7 家庭教育资本 `family_edu_capital`

父母学历分组完成后，取均值构建家庭教育资本综合指标，反映家庭背景对学业的系统性影响：

```
family_edu_capital = (Mother_qual_grouped + Father_qual_grouped) / 2
```

---

### Step 3 — 高基数变量分组

直接对高基数类别列做独热编码会产生极度稀疏矩阵，降低模型效果。采用**领域知识分组再独热**的两阶段策略：

| 原始列 | 原始类别数 | 分组后类别数 | 分组依据 |
|--------|-----------|-------------|----------|
| `Nacionality` | 21 种 | 5 组 | 地理区域（葡萄牙 / 欧洲 / 美洲 / 非洲葡语国 / 其他）|
| `Mother's qualification` | 29 种 | 5 组 | 教育层次（文盲→高等教育）|
| `Father's qualification` | 34 种 | 5 组 | 教育层次（同上）|
| `Mother's occupation` | 32 种 | 5 组 | 职业类别（无业 / 管理 / 专业 / 中级 / 蓝领）|
| `Father's occupation` | 46 种 | 5 组 | 职业类别（同上）|

---

### Step 4 — 去除冗余 & 合并稀疏类别

**删除 `International`（与国籍完全重叠）：**

`International` 与"非葡萄牙"标志的相关系数实测为 **1.000**，完全线性冗余。保留国籍分组（信息更丰富），删除 `International`。

**合并极稀疏 Application mode 类别：**

以下 5 个类别各自样本数 ≤ 10 条（正例率 < 0.25%），独热后经 StandardScaler 会产生 z-score ≥ 21 的异常值，对基于距离的算法造成极强干扰：

| 类别编码 | 含义 | 样本数 |
|---------|------|--------|
| 2  | Ordinance No. 612/93 | 3 |
| 10 | Ordinance No. 854-B/99 | 10 |
| 26 | Ordinance No. 533-A/99 (b2) | 1 |
| 27 | Ordinance No. 533-A/99 (b3) | 1 |
| 57 | Change of institution/course (International) | 1 |

处理方式：合并为 `Application mode = 99`（其他特殊渠道），共 16 条。

---

### Step 5 — 目标变量编码

使用 `LabelEncoder` 按字母顺序编码（非 OneHotEncoder，与 sklearn 分类器完全兼容）：

| 原始标签 | 编码 | 样本数 |
|----------|------|--------|
| Dropout  | 0 | 1,421 |
| Enrolled | 1 | 794   |
| Graduate | 2 | 2,209 |

同时生成**二分类版本** `Target_binary`（Dropout = 1，非 Dropout = 0），方便 Task 4 同时做三分类和二分类对比实验，无需重新运行预处理。

---

### Step 6 — 独热编码

**仅对名义类别列**执行独热编码，连续列和二值列不参与，避免极端 z-score 问题（旧版对所有列统一 StandardScaler 的核心问题）。

全部使用 `drop_first=True` 避免虚拟变量陷阱（完全多重共线性）：

| 列名 | 类别数 | 独热后列数 | 基准类（被 drop 的）|
|------|--------|-----------|---------------------|
| `Marital status` | 6 | 5 | 1（单身，占 88.6%）|
| `Application mode` | 14 | 13 | 1（第一志愿，占 38.6%）|
| `Course` | 17 | 16 | 33（生物燃料技术）|
| `Daytime/evening attendance` | 2 | 1 | 0（晚课）|
| `Nacionality_grouped` | 5 | 4 | 0（葡萄牙，占 97.5%）|
| `Mother_qual_grouped` | 5 | 4 | 0（文盲/未知）|
| `Father_qual_grouped` | 5 | 4 | 0（文盲/未知）|
| `Mother_occ_grouped` | 5 | 4 | 0（无业/未知）|
| `Father_occ_grouped` | 5 | 4 | 0（无业/未知）|
| `age_group` | 4 | 3 | 0（≤ 21 岁，应届生）|

---

### Step 7 — 数据集分割

采用**分层抽样**（`stratify=y`），保证三个类别在训练 / 验证 / 测试集中的比例完全一致：

| 集合 | 样本数 | Dropout | Enrolled | Graduate |
|------|--------|---------|----------|----------|
| 训练集（70%）| 3,096 | 32.1% | 18.0% | 49.9% |
| 验证集（10%）| 442   | 32.1% | 17.9% | 50.0% |
| 测试集（20%）| 886   | 32.2% | 17.9% | 49.9% |

分两步完成：先 70% / 30% 拆分，再将 30% 中的 1/3 作为验证集、2/3 作为测试集。

---

### Step 8 — 标准化（无数据泄漏）

**关键改进：Scaler 在分割之后才 fit，且仅在训练集上 fit。**

```python
scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])  # fit + transform
X_val[scale_cols]   = scaler.transform(X_val[scale_cols])        # 仅 transform
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])       # 仅 transform
```

**仅对连续 / 有序列标准化**（`nunique > 2`），共 **23 列**；二值列和独热列（`nunique ≤ 2`）保持 0/1 原值，共 **66 列**：

需标准化的 23 列包括：

- 学业表现（12 列）：两学期的 enrolled / evaluations / approved / grade / credited / without_eval
- 入学背景（3 列）：`Application order` / `Previous qualification (grade)` / `Admission grade`
- 宏观经济（3 列）：`Unemployment rate` / `Inflation rate` / `GDP`
- 派生特征（5 列）：`pass_rate_sem1/2` / `grade_trend` / `prev_qual_ordinal` / `family_edu_capital`

---

## 4. 与旧版 `dataclean.py` 的对比

| 检查项 | 旧版 `dataclean.py` | 新版 `data_preprocessing.py` | 改进说明 |
|--------|---------------------|-------------------------------|----------|
| **列名清洗** | 硬编码 `"Daytime/evening attendance\t"` | 读取后统一 `strip()` + `replace('\t', '')` | 消除列名污染，后续代码更健壮 |
| **年龄处理** | 直接标准化（连续值，最大值 70 岁）| 分为 4 个生命周期阶段，one-hot 编码 | 消除 441 个离群点的杠杆效应 |
| **grade = 0 处理** | 直接参与计算，当作真实 0 分 | 构建 `no_eval_sem1/2` 布尔标志 | 正确区分"缺考"与"真实零分" |
| **派生特征** | 无 | 新增 8 个（通过率 / 趋势 / 压力等）| 提升信息密度，增强模型可解释性 |
| **`Previous qualification` 处理** | 原始编码直接作为数值（顺序错误）| 按学历层次重映射为 0–4 有序整数 | 修正语义错误，编码顺序与学历对应 |
| **`International` 列** | 保留（与国籍高度重叠）| 删除（相关系数 = 1.000）| 消除完全冗余特征 |
| **Application mode 稀疏类** | 直接独热（产生 z-score=66）| 合并为"99-其他渠道" | 避免极端 z-score 干扰距离计算 |
| **StandardScaler 范围** | 对全部 85 列（含独热列）统一 StandardScaler | 仅对 23 个连续 / 有序列标准化 | 独热列保持 0/1，避免极端 z-score |
| **数据泄漏** | `scaler.fit_transform(X_all)` 在分割前执行 | 分割后在训练集 `fit`，其余仅 `transform` | 符合严格的无数据泄漏规范 |
| **目标变量** | 仅三分类编码 | 同时输出三分类 + 二分类（`Target_binary`）| 方便 Task 4 同时做两种实验 |
| **最终特征数** | 85 列 | **89 列** | 净增 4 列（删 2 增派生 8，独热合并减少）|

---

## 5. 输出文件说明

| 文件名 | 内容 | 用途 |
|--------|------|------|
| `full_preprocessed_data.csv` | 全量 4424 条，89 特征 + Target + Target_binary | t-SNE 可视化 / 聚类分析 |
| `train_70.csv` | 3,096 条训练集 | Task 4 模型训练 |
| `val_10.csv` | 442 条验证集 | Task 5 超参数调优 |
| `test_20.csv` | 886 条测试集 | Task 4/5 最终评估 |
| `standard_scaler.pkl` | 训练集 fit 的 Scaler 对象 | 新样本预测时复用 |
| `label_mapping.json` | `{"Dropout": 0, "Enrolled": 1, "Graduate": 2}` | 结果可解读 |
| `feature_names.txt` | 89 个特征名列表 | 特征索引参考 |
| `scale_cols.txt` | 23 个标准化列名 | 确认哪些列被标准化 |
| `preprocessing_log.txt` | 完整运行日志 | 调试 / 复现 |

---

## 6. 后续任务使用指引

### Task 2 — t-SNE 可视化

```python
import pandas as pd

df = pd.read_csv("full_preprocessed_data.csv")
X = df.drop(columns=["Target", "Target_binary"])
y = df["Target"]

# 推荐仅用高预测力连续特征，图形更清晰
core_features = [
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    "pass_rate_sem1", "pass_rate_sem2", "grade_trend",
    "Admission grade", "Tuition fees up to date", "economic_stress"
]
X_tsne = df[core_features]
```

### Task 3 — 聚类分析

```python
# 使用全量特征，全量样本（无需 train/test 区分）
X_cluster = df.drop(columns=["Target", "Target_binary"])
```

### Task 4 — 监督学习

```python
import pandas as pd

train = pd.read_csv("train_70.csv")
test  = pd.read_csv("test_20.csv")

X_train = train.drop(columns=["Target", "Target_binary"])
y_train = train["Target"]          # 三分类
X_test  = test.drop(columns=["Target", "Target_binary"])
y_test  = test["Target"]

# 类别不均衡处理：建模时传入
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
```

### Task 5 — 验证集使用

```python
val = pd.read_csv("val_10.csv")
X_val = val.drop(columns=["Target", "Target_binary"])
y_val = val["Target"]
# 用于超参数调优 / cross-validation
```

---

## 7. 环境依赖

```
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.5.0
joblib>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
ipykernel>=6.29.0
imbalanced-learn>=0.12.0
```

安装命令：

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyter ipykernel imbalanced-learn
```

---

## 8. 复现步骤

```bash
# 1. 激活环境
conda activate dsaa_ml

# 2. 确保 data.csv 与脚本在同一目录
ls data.csv data_preprocessing.py

# 3. 运行预处理
python data_preprocessing.py

# 4. 检查输出
ls *.csv *.pkl *.json *.txt
```

所有随机操作均设置 `random_state=42`，结果完全可复现。
