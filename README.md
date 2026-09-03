# SG-JEPA 与 RCPS-JEPA 对比实现

这是论文 *Scalable and Efficient Joint Spiking Embedding Predictive Architecture for Large-Scale Dynamic Graphs*（arXiv:2607.18412）的独立 PyTorch 复现。作者截至 2026-08-26 未公开官方代码；本文也未披露完整训练超参数，因此本仓库区分：

- **论文规定的实现**：非重叠时间窗口、GraphSAGE target encoder、RWPE、sinusoidal time encoding、PLIF、嵌套 spike count、共享前缀投影、learnable token/pooling、InfoNCE 与 target stop-gradient。
- **复现默认值**：隐藏维度、优化器、学习率、训练轮数等，集中在 `configs/sg_node_synthetic.yaml`，不声称是作者原始设置。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m jepa_compare.train_sg_jepa --config configs/sg_node_synthetic.yaml
pytest -q
```

## 通过 Git 交给其他机器运行

仓库跟踪 `data/processed/*.npz`，原始重复数据 `data/raw/` 不上传。处理后的
数据使用 Git LFS，因此上传者和运行者都必须安装 Git LFS。运行者克隆后只需
创建/激活 Python 3.10+ 环境、安装项目并执行统一入口：

```bash
git clone <repository-url> DYGJEPA
cd DYGJEPA
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

bash scripts/run_link_datasets.sh
```

如果运行者使用已有 Conda 环境，激活环境并执行 `python -m pip install -e .`
即可；运行脚本会优先使用仓库中的 `.venv`，否则使用当前环境的 `python3`。
也可以显式指定：

```bash
PYTHON_BIN=/path/to/python bash scripts/run_link_datasets.sh enron uci
```

快速 smoke test：

```bash
python -m jepa_compare.train_sg_jepa --config configs/sg_node_synthetic.yaml --epochs 1
```

## 数据格式

真实数据使用 `.npz`：

- `features`: `[T, N, F]`，所有 snapshot 共用全局节点 id；
- `edges_0` ... `edges_{T-1}`: 每个均为 `[2, E_t]` 的 COO edge index；
- `active`（可选）: `[T, N]` 布尔掩码；
- `labels`（可选）: `[N]`，仅用于冻结表征后的线性评估。

然后把配置中的 `data.path` 改为文件路径。论文没有说明 target 节点在历史 snapshot 缺失时的具体填充值；本实现要求其特征行存在，并通过 `active` 记录活跃性。若数据是事件流，应先离散化为论文采用的 snapshot 序列。

可在 `data` 下增加 `fanout: 5` 启用论文 Sec. 4.1 的固定邻居采样。RWPE 使用每节点 16 条随机游走估计返回概率，避免构造不可扩展的 `N x N` 稠密矩阵；游走数由 `model.rwpe_walks` 控制。

## DBLP 单数据集验证

### 1. 原始文件

将 MMDNE/SpikeNet 格式文件放到：

```text
data/raw/dblp/dblp.txt
data/raw/dblp/node2label.txt
```

`dblp.txt` 每行为 `source target normalized_timestamp`。数据包含 28,085 个
节点、236,894 条时间边、27 个时间步和 10 类；原始数据没有节点属性。

### 2. 快速 smoke test（不用于论文对比）

不提供特征文件时，转换器会生成 4 维结构替代特征：

```bash
python scripts/prepare_dblp.py
python -m jepa_compare.train_sg_jepa --config configs/sg_node_dblp.yaml
```

该模式只验证管线，结果通常接近多数类基线，不能和论文表格比较。

### 3. 生成论文对齐的 DeepWalk 特征

SpikeNet 官方代码调用：

```python
DeepWalk(80, 10, 128, window_size=10, negative=1, workers=16)
```

按构造函数顺序，它表示 80 维 embedding、walk length 10、每节点 128 次
随机游走、Word2Vec window 10、1 个 negative。**128 是游走次数，不是特征
维度**。最终 `dblp.npy` 的形状必须是 `[27, 28085, 80]`。

安装额外依赖：

```bash
pip install -e '.[features]'
```

使用官方参数生成全部 27 个累积快照特征：

```bash
python scripts/generate_dblp_deepwalk.py
```

该步骤约执行 9.7 亿次随机游走转移，在 CPU 上可能需要几十分钟到数小时，
并生成约 243 MB 的 `data/raw/dblp/dblp.npy`。可先用缩小参数检查环境，但
缩小结果不能用于论文比较：

```bash
python scripts/generate_dblp_deepwalk.py \
  --max-snapshots 1 \
  --walks-per-node 2 \
  --output /tmp/dblp-smoke.npy
```

检查正式特征：

```bash
python - <<'PY'
import numpy as np
x = np.load('data/raw/dblp/dblp.npy')
assert x.shape == (27, 28085, 80), x.shape
print(x.shape, x.dtype)
PY
```

生成 SG-JEPA 训练文件：

```bash
python scripts/prepare_dblp.py --features data/raw/dblp/dblp.npy
```

转换器会执行与 SpikeNet loader 一致的逐快照全局标准化，并拒绝把误写成
128 维的文件当成对齐特征。

### 4. 训练与评估

```bash
python -m jepa_compare.train_sg_jepa --config configs/sg_node_dblp.yaml
```

训练期间每隔 `checkpoint_every` 轮使用分层验证集训练轻量 MLP probe，按
验证集 Macro-F1 保存最佳模型到 `checkpoints/dblp_best.pt`。结束后会恢复最佳
checkpoint，在完整的 40% 标签配额上重新训练 MLP，并在独立测试集输出
`final_probe` 的 Macro-F1 和 Micro-F1。将 `train_ratio` 改为 `0.6` 或 `0.8`
可测试另外两个论文比例。

为避免节点 ID 排序造成类别偏置，InfoNCE target nodes 每轮随机打乱。RWPE
使用 `rwpe_seed + snapshot.time` 固定采样并缓存，同一快照在训练和推理阶段
使用完全相同的位置编码。`train_ratio` 是论文的总标签配额，其中默认 10%
用于 checkpoint 验证；最终测试前会把验证部分并回分类头训练集。

论文未公开完整 SG-JEPA 代码与全部最优超参数，所以此流程对齐已公开的数据
预处理和评估协议，但不能保证逐位复现论文数值。

## 代码与公式对应

| 论文部分 | 实现 |
|---|---|
| Eq. (5)-(6) temporal windows | `DynamicGraph.windows` |
| Eq. (7)-(9) target encoding | `encoding.py`, `GraphSAGE` |
| Eq. (10)-(11) PLIF/spike counts | `PLIF`, `forward_window` |
| Eq. (12) prefix projection | `SGJEPA.prefix_projection` |
| Eq. (13)-(14) predictor/pooling | `SGJEPA.predictor`, `pool_logits` |
| Eq. (15)-(16) InfoNCE/stopgrad | `SGJEPA.loss` |
| adaptive precision inference | `SGJEPA.infer(..., precision=k)` |

## 复现边界

论文报告 DBLP、Tmall、Patent，但没有给出预处理脚本、数据下载链接、节点特征构造方式及完整超参。当前版本因此完成了算法级复现和可运行验证框架；要逐表复现论文数值，还需要作者数据切分/预处理与缺失配置。生产训练建议把 RWPE 离线缓存，避免每轮重复随机游走。

## 动态链接预测：JODIE、DyRep、TGAT、DyGLib baselines 与 RCPS-JEPA

链接任务不再把 SG-JEPA 当作基线，因为原方法没有原生 link prediction 目标。主入口比较：

- `JODIELinkBaseline`：耦合 user/item RNN、elapsed-time trajectory projection、静态/动态 item 表示与 future-item scoring；
- `DyRepLinkBaseline`：点过程 intensity、结构注意力聚合、事件驱动递归更新和 sampled survival loss；
- `TGATLinkBaseline`：TGAT 官方仓库的 functional time encoding、递归 temporal convolution、20-neighbor sampling、2-layer/2-head edge-aware attention；
- `EdgeBankLinkBaseline`：Wikipedia 配置下的 unlimited-memory observed-edge lookup；
- `DyGLibLinkBaseline`：直接封装 DyGFormer 作者维护的统一库中 TGN、CAWN、TCL、GraphMixer 与 DyGFormer 源码，并与各方法原作者仓库逐项核对；
- `CLDGLinkBaseline`：CLDG 官方机制的无 DGL PyTorch 移植，保留 timespan-view sampling、GCN、projection MLP 和跨视图对称 InfoNCE；
- `MaskDGNNLinkBaseline`：activeness-aware temporal masking、归一化 GCN 与沿时间维的可学习 FFT filtering；
- `DVGMAELinkBaseline`：history-balanced temporal masking、variational GCN 和 temporal/global reconstruction decoder；
- `RCPSJEPA`：关系中心采样、二阶路径签名、Node–Relation JEPA、完整因果频次/事件语义记忆、方向敏感 user→page 表示、transductive ID residual、候选组排序损失、step-level EMA teacher 与离散 complementary-exponential link head；
- `jepa_compare.compare_link_prediction`：共享时间切分、正负边和随机种子，统一报告 AP、AUC、MRR 与 Recall@10。

当前 link comparison 支持 Wikipedia、MOOC、LastFM，以及 CanParl、Contacts、Flights、UNtrade、UNvote、USLegis、Enron、UCI，共 11 个数据集。前三个是有向二部交互数据；其余是同构事件流。比较脚本按目标时间划分 sliding windows，各模型只统一 train/validation/test 边界、query sampler、正负候选、指标和 validation-AP checkpoint rule；优化器、训练负采样、时间邻居采样和编码器仍按模型分别配置。二部图从 item 集合采样负目标，同构图从完整节点集合采样负目标。JODIE 的状态沿事件流连续推进；DyRep 保留 point-process likelihood 与 sampled survival 项；TGAT 保留作者代码的单负样本 BCE；TGN、CAWN、TCL、GraphMixer、DyGFormer 使用固定 DyGLib commit 的原始模块和各数据集对应超参；RCPS-JEPA 使用离散快照自监督目标。validation/test 均冻结网络参数，只允许因果状态推进或读取查询时刻之前的历史。

DyGLib 源码版本、许可证和本地差异记录在 `jepa_compare/dyglib_official/SOURCE.md`。模型文件除相对 import 外未改动；`jepa_compare/dyglib_baselines.py` 只负责快照到 event stream 的转换、padding id、共同 query 与指标。默认 event-stream 对照为 `edgebank, tgn, cawn, tcl, graphmixer, dygformer`，可通过配置中的 `additional_baselines.enabled` 选择子集；未加入 DTFormer 和 DySAT。

离散时间 SSL 对照由 `snapshot_ssl_baselines.enabled` 控制，默认启用 `cldg, maskdgnn, dvgmae`。三者只在 chronological training prefix 上做自监督预训练，之后冻结 encoder，并用完全相同的 PairMLP、训练 link queries、validation-AP checkpoint 和测试 candidate groups。这样得到的是 **SSL + frozen link probe** 表示比较，论文表格中必须与 JODIE/TGN/TGAT 等 native end-to-end link predictors 分组。实现来源和无法做到官方复现的边界记录在 `jepa_compare/snapshot_ssl_sources.md`。

DyRep 原论文没有在 Wikipedia 上报告结果，而且其数据包含 association/communication 两类事件。这里是明确的 **Wikipedia adaptation**：Wikipedia 只有一种 user–page communication stream，因此使用单 intensity channel，并从已观察交互构造动态邻域；不能把所得数值称为 DyRep 原论文复现值。TGAT 则直接参照用户指定的官方 ICLR 2020 仓库，在共同切分下覆盖其 Wikipedia 架构和训练默认值。

公平协议带来两个明确例外：官方 JODIE 评估对全部 item 排名并在 validation/test 流上继续反向更新参数；本比较改为所有模型完全相同的 1-positive/20-negative candidate groups，并冻结已选 checkpoint。时间差标准化和 `total_timespan / 500` 的 t-batch span 保留作者实现，state-label class weight 只由训练段估计，避免 validation/test label leakage。

当前 RCPS-JEPA 是与 SG-JEPA 输入格式直接兼容的 **snapshot 版本**：一个 snapshot 内的新增边被视为同一时间片事件，二阶 signature 对各时间片的关系统计增量编码。链接分支会从预测快照之前的完整离散历史读取重复 pair 次数与快照级 recency，但不会读取快照内连续时间顺序；JODIE、DyRep 和 TGAT 才使用逐事件时间戳。

### Wikipedia future-interaction prediction

这里使用 JODIE 的 Wikipedia 用户编辑页面数据。用户和页面使用不重叠的全局节点 ID；GNN 消息边去重并双向化，而 `query_edge_index` 单独保留所有有向重复编辑。`query_timestamps`、`query_features` 和 `query_labels` 与每条重复事件逐一对齐：JODIE 使用三者，TGAT 使用 timestamp 与原始 interaction features，DyRep 使用 timestamp，RCPS-JEPA 仍只使用离散 snapshot 表示且不读取 state-change label。

下载并转换：

```bash
mkdir -p data/raw/wikipedia
curl -L https://snap.stanford.edu/jodie/wikipedia.csv \
  -o data/raw/wikipedia/wikipedia.csv
python scripts/prepare_wikipedia.py
```

默认转换为 50 个等事件量快照。RCPS 的节点特征使用训练前缀统计归一化后压缩到 16 维，并加入节点类型、固定 identity、累计活跃度和 recency；`query_features` 单独保存原始 172 维 Wikipedia interaction vector，供 JODIE 与 TGAT 使用。修改预处理代码后必须重新运行转换命令。运行：

```bash
python -m jepa_compare.compare_link_prediction \
  --config configs/link_comparison_wikipedia.yaml
```

Link comparison 使用严格时间顺序的 75%/15%/10% 共同切分，`new_edges_only: false`，每个真实交互配 20 个只污染 destination 端的负样本。最终 JSON 包含 `jodie`、`dyrep`、`tgat`、配置启用的 DyGLib baselines、三个 snapshot SSL baselines 和 `rcps_jepa`。默认超参分别位于各模型的 `<model>_training` 段；命令行 `--epochs 1` 会把普通模型设为一轮，并把三个 SSL baseline 的 pretraining/probe 都设为一轮，仅用于 smoke test。

全部 11 个 link 数据集可以通过一个配置顺序运行。共同协议与训练规则只定义一次；配置中的 dataset entry 只覆盖数据类型、交互特征维度、模型兼容性，以及 DyGLib 官方代码报告的 dataset-specific best settings。JODIE 当前实现要求显式 user/item 分区，因此只在 Wikipedia、MOOC、LastFM 上运行；同构数据集不会伪造二部节点类型。

```bash
python -m jepa_compare.compare_link_prediction \
  --config configs/link_comparison_all.yaml

# 仍使用同一配置，只跑指定数据集
python -m jepa_compare.compare_link_prediction \
  --config configs/link_comparison_all.yaml \
  --datasets mooc lastfm
```

单轮端到端 smoke test：

```bash
python -m jepa_compare.compare_link_prediction \
  --config configs/link_comparison_wikipedia.yaml \
  --epochs 1
```

## 原始节点任务比较

节点任务采用与 SpikeNet / SG-JEPA 论文表格一致的最终快照、transductive 节点分类协议：完整累积图可见，标签按类别分层为 train/validation/test，所有方法使用完全相同的节点划分和 Macro-F1 checkpoint rule。主入口包含：

- `sg_jepa`：原版 SG-JEPA，冻结表示后训练 MLP probe；
- `rcps_jepa`：最佳 `hop2 + hop8` 多尺度冻结 probe；
- `evolvegcn_h`：官方 EvolveGCN-H 的 learned TopK、matrix-GRU 权重演化和两层 GCN，直接用节点标签训练；
- `roland`：GraphSAGE 各层表示作为 hierarchical node states，以 GRU 逐快照更新，并使用 ROLAND 的 truncated-BPTT 思路训练；
- `tgn`：官方 DyGLib TGN memory、message、time encoding 和 temporal attention backbone；
- `tgat`：官方 TGAT 的 harmonic time encoding、递归 temporal convolution 和 temporal-neighbor attention；
- `cawn`、`tcl`、`graphmixer`、`dygformer`：固定版本的 DyGLib 官方 backbone，以各自原生单负样本 future-link objective 预训练；
- `cldg`：timespan views 与对称 InfoNCE；
- `maskdgnn`：activeness-aware masking、时序 FFT filtering 与边重建；
- `dvgmae`：history-balanced masking、variational encoder 与 temporal/global reconstruction。

这里必须区分两组监督预算。`sg_jepa`、`rcps_jepa`、`tgn`、`tgat`、`cawn`、`tcl`、`graphmixer`、`dygformer`、`cldg`、`maskdgnn`、`dvgmae` 都是预训练时不读取节点类别的 **SSL + frozen probe**；其中 continuous-time 方法先做 future-link pretraining，三个 snapshot SSL 方法保留各自的自监督目标，之后全部冻结 encoder。`evolvegcn_h` 和 `roland` 是直接读取训练节点类别的 **supervised TGNN**。JSON 中的 `protocol`、`supervision` 与 `implementation` 字段会保留这个区别，论文表格也应分组呈现。

DBLP 只提供 27 个离散累积快照，没有逐边的原始精确时间。为运行 continuous-time 的 TGN/TGAT/CAWN/TCL/GraphMixer/DyGFormer，适配器从相邻快照做多重集合差得到新增无向边，保留重复交互，并把同一快照的边赋为相同时间；输出用 `snapshot_adapter` 明确标记。这是协议对齐的 DTDG→event adaptation，不应写成这些方法的官方数据复现。DBLP 的标签也不是 event-conditioned，因此 pair-conditioned backbone 最终用统一的 self-conditioned final-time node query 读出表示，JSON 会显式记录 `node_readout`。EvolveGCN-H、ROLAND 和三个 snapshot SSL 方法原生接收离散快照，不需要事件转换。

实现核对的官方 commit 为 EvolveGCN `90869062`、ROLAND `609f858d`、DyGLib `d55bbe67` 和 TGAT `66b2ccdd`。旧仓库依赖的 PyTorch/PyG 版本与当前环境不兼容，因此 EvolveGCN-H/ROLAND 在 `node_tgnn_baselines.py` 中逐式实现核心更新；TGN/CAWN/TCL/GraphMixer/DyGFormer 直接复用本仓库固定版本的 DyGLib 官方模块；TGAT 与 link baseline 共用已核对的核心递归和注意力模块。CLDG 是官方机制移植，MaskDGNN 与 DVGMAE 是论文级重实现，结果中不会误标为官方复现。

没有强行加入 `EdgeBank`、`JODIE` 和 `DyRep`：EdgeBank 不产生节点 embedding；JODIE 原始更新严格依赖 user–item 二部角色；DBLP 又没有 DyRep 所需的 association/communication 事件类型。把它们改造成最终快照的全节点分类器会改变方法定义，反而无法做到“尽量对齐”。CAWN 可以保留官方 pair encoder，并通过与 TCL/DyGFormer 相同的 self-conditioned readout 接入，所以予以保留并显式标记适配方式。

Synthetic：

```bash
python -m jepa_compare.compare_node_prediction --config configs/node_comparison_synthetic.yaml
```

DBLP：

```bash
python -m jepa_compare.compare_node_prediction --config configs/node_comparison_dblp.yaml
```

显存或时间有限时，用 `--baselines` 只运行指定 baseline；JEPA 两个主模型仍会保留：

```bash
# 只加两个原生 DTDG baseline
python -m jepa_compare.compare_node_prediction \
  --config configs/node_comparison_dblp.yaml \
  --baselines evolvegcn_h roland

# 不运行任何新增 baseline（兼容之前的两模型实验）
python -m jepa_compare.compare_node_prediction \
  --config configs/node_comparison_dblp.yaml \
  --baselines

# 只运行新增的离散快照 SSL baselines
python -m jepa_compare.compare_node_prediction \
  --config configs/node_comparison_dblp.yaml \
  --baselines cldg maskdgnn dvgmae
```

RCPS-JEPA 的 node branch 现在使用完整的关系条件节点表示：

1. 固定正交 `feature_skip` 保留 DeepWalk 输入语义，GraphSAGE 只学习带门控的结构创新；
2. `node_gru` 编码节点自身的连续历史创新量；
3. `node_relation_gru` 编码每个 snapshot 的邻居均值轨迹；
4. `temporal_node_increments` 显式记录度变化、加边、删边、邻居保留率和活跃状态；
5. 二阶 `truncated_signature` 保留局部结构事件的顺序；
6. `node_context_gate` 将关系更新以门控残差形式注入节点状态；
7. 残差动力学 predictor 保留当前节点语义，只预测未来创新量；
8. 唯一未来 node target 是 EMA encoder 的目标节点 latent，不再显式预测未来结构变化、完整邻域或新增邻居语义；
9. node batch 使用与 SG-JEPA 同类型的 InfoNCE，防止回归式 JEPA 丢失节点区分性；
10. `hop2 + hop8` 尺度一致读出对每个冻结尺度训练独立 MLP，并在固定测试前平均 logits，避免将局部与过平滑的长程信号提前压进同一向量；该最佳版本直接作为 `rcps_jepa` 输出。

node comparison 仍不读取候选 partner、节点类别或人工 link label，因此标签协议与 SG-JEPA 相同。训练日志输出 `contrastive_loss`、`graph_scale`、`history_scale`、`relation_gate` 和 `dynamics_scale`。候选 pair 的 Top-$B$ sampler、pair signature 和 link head 只在 link comparison 中启用。link relation target 也只由目标端点 latent 组合得到，不读取目标快照的 degree、common-neighbor、edge-arrival 等显式未来属性。
