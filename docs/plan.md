# Genres in Genres：项目叙事与推进计划

## 1. 核心问题（要讲的数据故事）
**问题一句话版**：一个艺术家的音乐风格会如何随时间演化？这种演化是连续漂移，还是存在明显阶段？在“主流 genre”之下，能否发现更细粒度、可解释的“子风格（genres in genres）”？

将问题拆成 3 个可回答子问题：
1. **宏观演化**：不同专辑/时期之间风格差异有多大？哪里发生“跳变”？
2. **专辑内部**：同一张专辑是“统一风格”还是“拼盘式多样”？
3. **子风格结构**：全职业生涯能否聚成若干个子风格簇？这些簇能否用人类可读的语义标签解释？

## 2. 数据与准备（数据思维的起点）
- **数据来源**：本地音乐库音频文件，组织为 `data/music/{Artist}/{Year}-{Album}/*.(mp3|wav)`（`src/library_manager.py`）。
- **样本单位**：track（歌）；聚合单位：album（专辑/时期）。
- **特征表示**：每首歌用 MuQ‑MuLan 抽取 512 维 embedding（`src/extractor.py`），统一截取音频中间 30s 以保证可比性。
- **语义词表**：从 `data/metadata/id_tags.csv` 统计高频 tag，编码为文本 embedding，用于解释音频簇（`src/semantics.py`）。

## 3. 方法与指标（把“风格”变成可计算对象）
- **降维可视化**：PCA / t‑SNE / UMAP（`StyleAnalyzer.reduce_dimension`）。
- **聚类发现子风格**：KMeans（`StyleAnalyzer.cluster_songs`）；可选自动选 K：Silhouette（`find_optimal_k`）。
- **三大指标（支撑结论）**：
  - `velocity`：相邻专辑中心的余弦距离，衡量“专辑间变化幅度”（`MusicMetrics.calculate_style_velocity`）。
  - `novelty`：当前专辑 vs 历史均值中心的余弦距离，衡量“背离过去自我”（`calculate_novelty`）。
  - `cohesion`：1 − 专辑内平均离散度，衡量“专辑一致性/多样性”（`calculate_cohesion`）。

## 4. 叙事结构（报告/展示建议顺序）
1. **动机与问题**：为什么“风格演化”值得研究？为什么“子风格”比大类 genre 更有信息量？
2. **数据准备**：说明数据结构、抽样策略（中间 30s）、缺失与限制。
3. **方法框架**：embedding 作为“风格坐标系”，聚类作为“子风格发现”，语义映射作为“可解释性桥梁”。
4. **回答子问题（每张图对应一个问题）**：
   - 轨迹图：宏观演化与跳变点
   - Cohesion：每张专辑内部统一/多样
   - Streamgraph + Composition：子风格随专辑演化、簇与专辑的对应关系
   - Radar：关键专辑的语义维度对比（从“抽象向量”回到“人能读懂的词”）
5. **证据闭环**：在 UI 的 Explorer 里点播几首歌，验证“簇的听感一致”。
6. **结论与反思**：给出 2–3 条结论（“哪张最创新”“哪里转型”），并说明方法与数据的局限。

## 5. 可视化与“回答什么问题”的对应关系
- `Trajectory`（`src/visualization.py::plot_2d_trajectory`）：回答“宏观演化是否连续/是否跳变”。
- `Consistency`（`plot_consistency`）：回答“哪张最统一、哪张最杂”。
- `Streamgraph`（`plot_streamgraph`）：回答“子风格簇如何随专辑更替”。
- `Cluster Composition`（`plot_cluster_composition`）：回答“每个簇主要出现在哪些专辑”。
- `Radar`（`plot_radar`）：回答“关键专辑在语义维度上如何变化”。
- `Explorer + 播放`（`app.py` 动态渲染）：回答“结果是否可听、可验证”。

## 6. 已完成（基于现有 codebase）
- 扫描→抽取→缓存：`scripts/preprocess.py` + `src/library_manager.py` + `src/extractor.py`。
- 核心分析：降维、聚类、自动选 K、报告生成：`src/analysis.py`。
- 指标计算：`src/metrics.py`（velocity/novelty/cohesion）。
- 可解释性：`src/semantics.py`（从 `id_tags.csv` 构建 tag 词表并做最近邻标注）。
- 交互呈现：`app.py`（多图联动、参数可调、簇内曲目表与播放）。

## 7. 待完成（按优先级）
1. **选定主案例艺术家 + 2–3 个“剧情点”**：例如“转型专辑/实验期/巅峰期”，在展示时固定讲这几个点。
2. **补数据概况**：每张专辑 track 数、年份跨度、缺失情况（作为“先看数据长什么样”）。
3. **给聚类最小可靠性证据**：展示 silhouette 随 K 的变化，或多次随机种子下聚类是否稳定（不追求论文级，但要有“不是拍脑袋”）。
4. **术语与文案统一**：cohesion/variance 的口径统一，避免“越大越好/越大越差”混乱。
5. **解释标签机制**：明确 tag 是“文本与音频 embedding 相似度”，不是人工标注，避免误解。
6. （可选加分）**阶段检测**：基于专辑中心距离序列做简单变点/峰值，输出“第 1/2/3 阶段”。

## 8. 交付物（建议）
- `docs/report.md` 或 `docs/slides.md`：按第 4 节叙事结构撰写，图表截图来自 app。
- 一段 3–5 分钟演示脚本：固定顺序跑 `Library → Insight Report → Explorer`，用 2–3 首歌做听感验证。

