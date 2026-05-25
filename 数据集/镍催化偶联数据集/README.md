# 镍催化不对称交叉偶联反应数据集

来源：AI-Driven Development of Nickel-Catalyzed Enantioselective Cross-Coupling Reactions

## 文件说明

### 主数据集

| 文件 | 用途 |
|---|---|
| `Raw_Dataset.csv` | **YONOD 直接使用的主文件**，含 SMILES 和预测目标 |

**列说明（Raw_Dataset.csv 核心列）：**
- `Ligand_SMILES`：催化剂配体的 SMILES
- `Product_SMILES`：产物底物的 SMILES
- `Temperature`：反应温度（°C）
- `△△G (Kcal/mol)`：预测目标，两对映体过渡态自由能差（kcal/mol）
- `ee (%)`：对映体过量值，与 ΔΔG 等价（`ee = tanh(ΔΔG/2RT) × 100`）

数据规模：**6590 条**有效反应记录

---

### 预计算描述符/

用于与 YONOD 的 SMILES 路线做基线对比，**不含 SMILES 列**，需与 Raw_Dataset.csv 按行对应使用。

| 文件 | 维度 | 内容 |
|---|---|---|
| `DFT描述符_93维.csv` | 93 + 1（温度）| 量子化学描述符（HOMO/LUMO、偶极矩、Mulliken电荷等），原论文最优特征集，**需要量子化学计算** |
| `RDKit描述符_239维.csv` | 239 + 1（温度）| RDKit 物理化学描述符（logP、TPSA、旋转键数等），可实时计算 |
| `配体DFT描述符.csv` | 若干 | 仅配体的 DFT 描述符，按配体归类 |
| `产物DFT描述符.csv` | 若干 | 仅产物的 DFT 描述符，按产物归类 |
| `电子能量.csv` | 若干 | 各分子的电子能量原始数据 |

---

### 文献/

| 文件 | 内容 |
|---|---|
| `原论文.pdf` | 原始研究论文 |
| `原论文.md` | 论文 Markdown 版（便于检索引用） |

---

## 未纳入的文件

以下文件**未复制**，如需要可从原始目录获取：

| 文件 | 原因 |
|---|---|
| `Data_AutoGluon_MAF.csv`（13MB） | MAF 嵌入（需要专用模型生成），暂不使用 |
| `Ligands_DFT_Coordinate.csv`（1.3MB）| 3D 坐标数据，ML 训练不直接使用 |
| `Prodcuts_DFT_Coordinate.csv`（10MB）| 同上 |
| `Data\xlsx\*.xlsx` | 与 csv 内容相同，保留 xlsx 原件在原目录 |

原始文件位置：`数据集/Enantioselective-Cross-Coupling-Prediction/`
