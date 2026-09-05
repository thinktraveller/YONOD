# ChemSpace Builder 项目解读与 YONOD 对接建议

核查日期：2026-09-05。依据本地 `reference-proejct/chemspace_builder/` 代码和 YONOD 当前工作区；未修改两个项目的业务代码。本文区分现有功能与建议扩展。

## 1. 可以用来做什么

这是一个 Streamlit 化学空间可视化应用：上传多个含 SMILES 的 CSV，指定化学角色和列，计算分子特征，降维后用不同颜色/形状展示不同数据集，导出 PNG（300 dpi）、SVG、坐标 CSV。

适合探索底物或产物结构覆盖范围、比较建模集和外部验证集、制作论文配图。它本身不训练产率预测模型、不执行交叉验证，也不自动验证数据集在化学组分上独立。CSV 中已有的 predicted 列不是它预测出来的。

主要文件：

| 文件 | 用途 |
|---|---|
| `reference-proejct/chemspace_builder/app.py` | 约 1180 行单文件应用，包含向导、特征计算、降维和导出 |
| `reference-proejct/chemspace_builder/environment.yml` | Python 3.10 Conda 环境声明 |
| `reference-proejct/chemspace_builder/draw_fig.ipynb` | 从四个原始 CSV 计算三组分反应特征、运行 t-SNE 并保存图的示例；不是只读取已导出的坐标 |
| `reference-proejct/chemspace_builder/dataset/` | 四组示例数据，列含 dioxazolone、OH、product、yield (%)、predicted |

## 2. 代码实际如何工作

### 普通路径：一个点是一个分子实例

`build_points_from_roles()`（app.py:284）将各选中列展开成长表。同一反应选三列通常得到三个分子点；相同分子多次出现不会自动去重。界面中的 Whole reaction 在这条路径表示“选择全部角色列”，不表示“拼接成一个反应向量”。

`featurize_smiles()`（app.py:160）支持 MorganFP、RDKit descriptors、两者拼接、再加 MACCS。默认 Morgan 为 radius=2、2048 位；MACCS 为 167 位且需同时选择包含 MACCS 的模式并勾选 Include MACCS。普通 RDKit 路径使用 `Descriptors._descList`。

### 特殊路径：一个点才是一条反应

app.py:925 的隐藏分支要求同时满足：

- Whole reaction；
- t-SNE；
- RDKit + MorganFP + MACCS，并勾选 Include MACCS；
- Morgan 2048 位、radius=2。

此时仅取 Substrate 1、Substrate 2、Product 各自映射的第一列，拼接三个分子特征块；Substrate 3 和 Other 不进入这条反应特征路径。任一组分非法即删除整条反应。RDKit 特征改用 `rdMolDescriptors.Properties()`，与普通路径的描述符集合也不同（app.py:333、380）。

### 降维与绘图

`embed_2d()`（app.py:231）对所有输入点统一执行 StandardScaler，再运行 PCA、UMAP 或 t-SNE。UMAP 固定 cosine 距离；t-SNE 可选 euclidean/cosine。t-SNE 默认实际计算三维，后续仅取前两维绘图，不能称为直接优化的二维 t-SNE。

所有上传数据联合嵌入，适合描述性总览。若未来把降维结果用于模型评估，必须改为训练折拟合缩放器和可变换的降维器、验证折仅 transform，不能复用全数据拟合结果。UMAP 官方提供此流程：[Transforming New Data](https://umap-learn.readthedocs.io/en/latest/transform.html)。

图形主体由 Matplotlib/Seaborn 绘制（app.py:1072 之后）。虽然强制导入 Plotly，但当前没有实际的 Plotly 交互散点图调用。`st.cache_data` 缓存读取、普通分子特征和降维；`cached_points` 保存坐标，使颜色、尺寸、图例等样式调整不需要重新计算。

## 3. 与 YONOD 的关系

YONOD 负责比较描述符和模型对反应目标的预测表现；ChemSpace Builder 可为其补充特征空间与数据覆盖的展示层。它没有现成的 YONOD 专用适配器，也不会自动读取 YONOD 的 NPZ 产物或交叉验证预测。

当前 YONOD 已超出早期 README 的 4×4 原型，包含多描述符注册、持久化特征、独立 benchmark、重复交叉验证与报告路径。对接应以实际代码为准。

| YONOD 现有能力 | 推荐的对接方式 |
|---|---|
| `yonod/universal/descriptor_artifact.py:189`：读取特征产物 | 直接对 `X_smiles` 降维，保证与 YONOD 的描述符实现一致 |
| 产物的 sample_ids、valid_mask | 坐标对应 `sample_ids[valid_mask]`；不能与完整 ID 数组直接拼接 |
| `yonod/universal/feature_builder.py:249`：逐列 concat | 一行反应对应一个点；明确记录输入列、顺序、描述符和模式 |
| `yonod/splits/manifest.py:24`：切分清单 | 选择一个 repeat/fold，按 train/validation 着色；repeated_kfold 的 group_id 不是化学分组 |
| `yonod/benchmark/executor.py:329`：保存验证预测 | 连接真实值、预测值和残差，探索误差集中在哪些结构区域 |
| `yonod/benchmark/report.py:455`：报告生成 | 增加空间图、坐标表和参数记录；通过 layout.py 适配新旧输出目录 |

建议的最小扩展流程：描述符 NPZ → 有效样本 ID 对齐 → PCA/UMAP/t-SNE → 连接切分或预测 → 生成空间图和 CSV → 嵌入现有报告。这是建议设计，尚未实现。

数值辅助列和标签不在 `X_smiles` 内；如果要解释完整模型输入，需另行按同一 mask 对齐连接。OHE 在训练折拟合，也不能假设存在统一的全数据描述符 NPZ。

重复交叉验证中，同一样本有多次验证预测；误差着色必须选择描述符、模型、repeat，或明确定义跨 repeat 聚合，防止连接后重复展开。

## 4. 哪些值得借鉴，哪些需要先调整

最值得借鉴：

1. 多 CSV 的数据集命名、颜色和标记配置，便于比较建模集、外部集及不同反应数据来源。
2. 四步向导和角色映射交互，让数据选择、特征选择、降维参数、出图相互分开。
3. 缓存坐标后独立调整图形，避免每改一次图例都重算嵌入。
4. PNG/SVG 与坐标表同时导出，可继续修改图形或对接其他分析。

接入前应调整：

- 显式区分“分子点”和“反应点”，去掉由算法参数隐式切换数据语义的行为。
- 直接复用 YONOD 描述符。YONOD Morgan 默认为 radius=2、1024 位，而 mfp 为 radius=3、1024 维计数指纹；本项目默认 Morgan 为 2048 位二值指纹。YONOD MACCS 去除占位 bit 后为 166 维。不能仅按名称认定一致。
- 统一缺失处理：本项目普通路径丢弃非法分子，特殊路径任一组分非法即丢整行；YONOD 普通 concat 通常给非法组分零块，并保留至少一列有效的行，MFP 还有自己的保留策略。
- 不勾选当前的 Keep invalid SMILES rows：它的说明声称不参与嵌入，但实际勾选后零向量仍会送进降维（app.py:1018）。
- 保留稳定 sample_id、真实值、预测值、描述符参数、算法参数、seed、依赖版本和过滤记录。当前导出仅有来源、角色、SMILES、row_index、有效标记及坐标，并未像 README 所述完整携带所有原始列。
- 对大量数据增加抽样或去重，并说明统计单位。当前普通分支固定 `sample_per_dataset=0`；数万反应乘多个角色会展开成大量分子实例。
- 为 t-SNE 校验 `perplexity < 有效点数`；当前默认 50，单独使用 External_8.csv 的小样本时会失败。[scikit-learn TSNE 文档](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

二维图用于辅助解释；簇分得开不等于描述符预测能力更好。对 YONOD 的模型优劣结论仍应来自一致的交叉验证、外部评估和误差统计。产物是否进入空间应按研究问题显式决定；如果要解释一个没有使用产物结构的模型，图也应采用相同输入范围。

## 5. 如何启动

本机已经存在 `C:\ProgramData\anaconda3\envs\chemspace`。优先复用，不执行 README 中可选的环境删除命令。

实际验证结果：Python 3.10.20、RDKit 2026.03.3、Streamlit 1.59.0、UMAP 0.5.12、scikit-learn 1.7.2 及其余绘图库均成功导入；Streamlit AppTest 加载 app.py 首页，异常列表为空，显示 `1) Data`。沙箱内首次导入阻塞在 Numba 的缓存路径写入，获准在沙箱外检查后通过。验证范围为依赖导入和首页脚本执行，没有完成浏览器上传、全量降维和下载测试，也没有留下常驻服务。

在支持 Conda 的终端运行：

```powershell
Set-Location 'D:\大创\YONOD\reference-proejct\chemspace_builder'
conda activate chemspace
python -m streamlit run app.py --server.address 127.0.0.1
```

如果 PowerShell 中 conda activate 未初始化，可直接指定已有解释器：

```powershell
& 'C:\ProgramData\anaconda3\envs\chemspace\python.exe' -m streamlit run 'D:\大创\YONOD\reference-proejct\chemspace_builder\app.py' --server.address 127.0.0.1
```

通常访问 http://127.0.0.1:8501；若默认端口占用，以终端显示的地址为准，或增加 `--server.port 8502`。终端中 Ctrl+C 停止服务。

仅在另一台机器没有 chemspace 环境时，从该项目目录执行：

```powershell
conda env create -f environment.yml
conda activate chemspace
python -m streamlit run app.py --server.address 127.0.0.1
```

### 第一次操作建议

1. 上传自带四个 CSV。可按 notebook 将 Model_480、StrictIndependent_180、Alcohol_96、External_8 分别命名为 modeling、component-disjoint、class-shift、prospective；这些是示例标签，不代表应用验证了对应划分性质。
2. 映射 Substrate 1=dioxazolone、Substrate 2=OH、Product=product。`substrate` 列实际是 A1-B1 这类编号，不是 SMILES。
3. 先用 Select role-columns 只选一个底物列、MorganFP、PCA，清楚观察“一个点一个分子实例”的普通流程。
4. 进入 Plot & Export，点击 Build & plot chemical space，再调整图例并下载。
5. 若要复现 notebook 的反应空间，使用第 2 节全部特殊路径条件，并保留默认三维 t-SNE 及其参数；若要直接做二维 t-SNE，改 n_components=2，但图形将与 notebook 不同。

用 YONOD 酰胺缩合数据做初步底物探索时，可以选择 sub_1_smiles 或 sub_2_smiles。要展示与 YONOD 模型一致的完整反应空间，采用第 3 节的 NPZ 对接方案更直接；现有界面没有这一入口。
