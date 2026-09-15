# Findings & Decisions

## Requirements

从用户请求中提取的明确需求：

- 使用 `planning-with-files` 进行项目规划，规划内容需持久化到文件。
- 当前目标已收窄：仅做人脸识别，跑通成熟开源人脸识别模型，作为后续预训练/特征提取 baseline。
- 使用开源识别模型。
- 通过 SSH 连接远程服务器进行部署、训练或运行。
- 暂不构建向量数据库。
- 暂不做猫脸识别。
- 当前服务器信息：`49.232.174.36:2006`，用户 `cx`，密码登录。密码不写入文件。

## Research Findings

### 开源人脸识别路线

- InsightFace 是成熟的开源 2D/3D 人脸分析项目，覆盖人脸检测、对齐、识别等流程，适合作为人脸系统复刻 baseline。
- ArcFace 是人脸识别中非常经典的 embedding/度量学习方法，适合“特征向量 + 相似度检索 + 阈值判定”的系统形态。
- 人脸识别 MVP 可以先用现成预训练模型跑通，不必一开始重新训练。
- 远程服务器已确认：Ubuntu 24.04.3 LTS，NVIDIA GeForce RTX 3060 12GB，驱动 580.159.03，/home 可用空间约 3.1TB。
- 系统 Python 为 3.13.x；为了兼容视觉模型依赖，建议新建 Python 3.10/3.11 的 Conda 环境。

### 向量数据库路线

- Qdrant 是开源向量数据库/向量搜索引擎，支持向量、payload、过滤和相似度检索；MVP 部署简单，适合先落地。
- Milvus 是高性能、可扩展的开源向量数据库，适合更大规模或分布式向量检索场景。
- 本项目第一阶段建议默认 Qdrant；如果未来猫脸图片/embedding 达到百万级、需要复杂运维和分布式扩展，再评估 Milvus。

### 猫脸/动物个体识别路线

- 公开资料显示已有 PetFace 等大规模动物脸个体识别数据集/benchmark，覆盖猫、狗、兔等多个动物类别。
- Kaggle 上存在 Cat Individual Images 数据集，公开描述为 518 只猫、13,536 张图，可作为猫脸识别 baseline 的候选数据来源之一。
- 斯坦福 CS230 项目曾做 Pet Cat Face Verification and Identification，公开报告描述过 23,500+ 只猫、130,000+ 张图级别的数据集实验。
- 猫脸识别不应简单等同于猫/非猫检测；目标是区分 Cat A、Cat B、未知猫，这属于更细粒度的个体识别/开放集识别问题。
- 猫脸比人脸更容易受到姿态、毛发、遮挡、光照、成长变化、同品种相似度影响，因此数据质量和人工复核闭环非常关键。

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| 当前只跑通人脸识别 baseline | 用户明确调整范围，先把成熟人脸模型跑起来。 |
| 人脸 baseline 使用 InsightFace/ArcFace | 成熟、开源、资料多、可直接产出 embedding。 |
| 首次远程运行优先 CPU 版 onnxruntime 兜底 | GPU/CUDA 状态未知，CPU 版更容易先完成 smoke test。 |
| 新建独立 Conda 环境 | 避免系统 Python 3.13 与视觉依赖的兼容问题。 |

## Candidate Model and Data Options

### Human face MVP

- Model/toolkit: InsightFace。
- Detector: SCRFD 或 RetinaFace。
- Embedding: ArcFace 系列模型。
- Inference API: FastAPI。
- Vector DB: Qdrant first, Milvus optional.

### Cat face MVP

- Detector:
  - 方案 A：先用 YOLO 系列微调猫脸检测。
  - 方案 B：寻找公开猫脸 landmark/detection 模型作为起点。
- Recognition:
  - 方案 A：ResNet/EfficientNet/ViT backbone + ArcFace/CosFace。
  - 方案 B：参考 PetFace/动物个体识别 benchmark。
  - 方案 C：借鉴 DogFaceNet 类度量学习结构迁移到猫脸。
- Dataset:
  - 公开猫个体数据集。
  - 自采猫脸数据。
  - 用户登记数据经复核后回流训练。

## Suggested Vector Schema

### Collection: `human_faces`

```json
{
  "id": "uuid",
  "vector": "float[512]",
  "payload": {
    "subject_id": "person_xxxx",
    "species": "human",
    "image_id": "img_xxxx",
    "embedding_model": "insightface_arcface_xxx",
    "quality_score": 0.0,
    "source": "upload|camera|batch",
    "created_at": "ISO-8601",
    "label_status": "confirmed|pending|rejected"
  }
}
```

### Collection: `cat_faces`

```json
{
  "id": "uuid",
  "vector": "float[512 or model_dim]",
  "payload": {
    "cat_id": "cat_xxxx",
    "species": "cat",
    "image_id": "img_xxxx",
    "embedding_model": "cat_arcface_v1",
    "quality_score": 0.0,
    "pose": "front|left|right|unknown",
    "source": "upload|camera|batch",
    "created_at": "ISO-8601",
    "label_status": "confirmed|pending|rejected",
    "review_required": false
  }
}
```

## Evaluation Plan

- 人脸 MVP：
  - 小样本 sanity check：同一人多图相似度高于不同人。
  - Top-K 检索结果正确。
  - 阈值可配置，结果包含相似度。
- 猫脸模型：
  - Verification: AUC、EER、TAR@FAR。
  - Identification: Top-1、Top-5、mAP。
  - Open-set: 未知猫拒识率、误识率。
  - 业务指标：登记成功率、重拍率、人工复核率、误合并率、误拆分率。

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| 使用 `/home/cx/train/9/*.png` 做人脸 smoke test 时检测到 0 张脸 | 推断这些图片不是人脸样本；InsightFace 初始化正常，需更换真实人脸图片继续验证 |

## Latest Batch Evaluation (v3)

- 数据集共有 85 张照片；排除单图组并使用每人前 3 张登记后，共得到 9 个人、53 张测试照片。
- 参数：`threshold=0.50`、`enroll-count=3`、`exclude=test`、`min-images=2`。
- 结果：48/53 正确，accuracy=0.9057；48 张 matched，3 张 unknown_person，2 张 no_face。
- 失败照片：`hb7.jpg`、`xzh8.jpg`、`zr5.jpg`、`zxy4.jpg`、`zxy5.jpg`。
- `hb7.jpg` 是横向躺姿且脸部接近侧面；尝试旋转 0/90/180/270 度仍无法检测到脸。
- `zxy5.jpg` 是明显侧脸，模型无法检测；`xzh8.jpg`、`zr5.jpg`、`zxy4.jpg` 均有低头或较大侧转，相似度分别为 0.446751、0.464069、0.348025。
- 当前不建议为了这几张困难照片继续降低阈值。降低阈值可能提高这组已知人员测试的分数，但也会增加把陌生人误认为已登记人员的风险。
- 当前最有价值的下一步是理解并整理核心命令行流程。网页只是可选操作界面；小规模人员数据继续使用本地 NumPy 文件即可，向量数据库留到人数和 embedding 数量明显增长后再接入。

## Project Audit (2026-09-02)

- 当前代码不是严格的一键运行：需要先进入服务器项目目录、激活 `face310` 环境，再分别运行 demo、登记、识别或评估脚本。
- 核心链路位于 `scripts/face_demo.py`：`load_image` 读取图片，`build_app` 加载 InsightFace，`analyze_image` 检测并提取 embedding，`cosine_similarity` 比较两个 embedding。
- 登记链路位于 `scripts/register_person.py`：复制一张照片、提取一个 embedding、保存 `embeddings.npy`、更新 `registry.json`。
- 识别链路位于 `scripts/recognize_person.py`：读取登记表、提取查询图片 embedding、遍历已登记人员、取最高相似度、用阈值决定匹配或未知。
- `scripts/evaluate_grouped_faces.py` 和 `scripts/sync_face_dataset.py` 是评估/数据整理工具，不是日常识别核心。
- `scripts/web_app.py` 是可选操作界面，不是核心算法；当前路线已暂停网页工作。
- 与最初目的的关系：人脸 baseline 已完成；向量数据库和猫脸识别尚未完成，这是用户主动把当前阶段收窄后的预期结果，不是意外偏差。
- 人脸模型可以作为工程流程参考，但不能直接当作猫脸识别模型。猫脸需要动物脸数据、检测/对齐方案和后续训练或度量学习实验。
- 当前仍有工程欠缺：登记脚本一次只接收一张照片；识别和登记依赖从项目根目录运行；多脸图片默认取最大脸；远程使用 CPU provider，RTX 3060 尚未用于推理加速。

## Server Outage Update (2026-09-03)

- 用户确认远程服务器近期崩溃且无法连接。
- 服务器状态、远程 Conda 环境和远程项目文件是否完整尚无法确认；不应将历史运行成功等同于当前仍可用。
- 本地仍保留核心脚本、分组照片和 v3 评估报告，因此可继续代码学习、工程整理和报告分析。
- 当前进度的准确定位是“预训练模型集成 baseline 已验证”，而不是“已完成可稳定部署的系统”。
- 用户随后反馈服务器已修复；实际恢复验收显示 `120.48.24.192:2006` 可建立 TCP 连接，但 SSH 会话在认证前被远程关闭。
- 命令行显示 `Connection closed by 120.48.24.192 port 2006`，Termius 显示 `Connection closed with error: end of file`；因此当前状态为“主机/端口已恢复，SSH 服务尚未通过验收”。

## Server Recovery Verification (2026-09-03)

- 服务器新地址为 `49.232.174.36:2006`，Termius 已成功以 `cx` 登录。
- 系统仍为 Ubuntu 24.04.3 LTS，`/home` 使用约 15.4% / 3.58TB。
- NVIDIA GeForce RTX 3060 12GB 可见，驱动版本 580.173.02。
- `/home/cx/face_recognition_baseline`、`/home/cx/miniconda3`、`/home/cx/.insightface` 均存在；核心脚本、样例图片、分组数据、登记表和历史报告仍在。
- `face310` 环境可激活，Python 版本为 3.10.20，InsightFace/ONNX Runtime/OpenCV/NumPy 已安装。
- 用 `data/samples/astronaut.jpg` 运行 `face_demo.py` 成功：`faces_detected=1`、`det_score=0.8364`、`embedding_dim=512`。
- InsightFace 当前仍对所有 buffalo_l ONNX 模型使用 `CPUExecutionProvider`；GPU 硬件可见不等于 ONNX Runtime CUDA provider 已可用。

## Unified CLI (2026-09-03)

- 2026-09-04 已将 `face_cli.py`、`README.md` 上传服务器，SHA-256 与本地一致；统一入口真实单图推理验收通过（1 张脸，512 维 embedding）。仍需先激活 `face310`；自动选择环境尚未实现。

- 新增根目录 `face_cli.py`，统一提供 `demo`、`register`、`recognize`、`evaluate` 四个子命令。
- 统一入口使用 `sys.executable` 调用原有脚本，确保继续使用当前激活的 Conda/Python 环境。
- 默认数据和报告路径从 `face_cli.py` 所在的项目根目录解析；图片等用户输入路径从发起命令的当前目录解析。
- 本次保留原有脚本不变，因此原脚本直接运行时仍可能依赖当前目录；底层路径改造尚未完成。
- 新增 `README.md`，记录环境、四个统一命令、数据流、推荐评估参数和已知限制。

## Resources

- InsightFace GitHub: https://github.com/deepinsight/insightface
- InsightFace ArcFace research overview: https://www.insightface.ai/research/arcface
- Qdrant documentation: https://qdrant.tech/documentation/
- Qdrant GitHub: https://github.com/qdrant/qdrant
- Milvus documentation: https://milvus.io/docs
- Milvus GitHub: https://github.com/milvus-io/milvus
- PetFace project page: https://dahlian00.github.io/PetFacePage/
- PetFace arXiv page: https://arxiv.org/html/2407.13555v1
- Cat Individual Images dataset: https://www.kaggle.com/datasets/timost1234/cat-individuals
- Pet Cat Face Verification and Identification report: https://cs230.stanford.edu/projects_fall_2019/reports/26251543.pdf

### 2026-09-09 猫脸归档初检

- `/home/firecom/yjr/cat_data/cat.tar.gz` 约 45 GB，原始归档保持只读；服务器 `/home` 剩余空间约 2.9 TB。
- 归档顶层为 `cat/`，样例路径为 `cat/038678/00.png`；数字子目录极可能是个体 ID，PNG 是主要样本格式。
- 完整列目录扫描需要顺序解压读取 45 GB gzip，因此时间较长属正常。应先利用该结果排查绝对路径和 `..` 路径，再考虑解压。
- 完整清单包含 643,539 张 PNG 和 164,100 个体，每个体 2–10 张、中位数 4；这是典型的大类别、小样本个体识别数据，更适合 embedding/度量学习和 retrieval 评估，不适合直接做 164,100 类的普通封闭集分类作为最终方案。
- 服务器已存在 `cat-recognition-system`：数据脚本会流式读取 tar.gz、计算内容哈希找完全重复，并依据 CSV 将整个身份分配到单一 split，这可避免同一只猫同时出现于训练和测试集的泄漏。
- 现有 split 为 80/10/10 左右：train 131,358 个身份，validation 16,378，test 16,364；因为测试身份与训练身份不重叠，可用于更严格的开集猫个体检索评估。
- 完全重复审计发现 16,119 个内容哈希出现在多个身份中；相关的 40,628 张图已全部排除。这不只是存储去重，而是防止标签冲突与评估泄漏的必要清洗。
- Frozen DINOv2 ViT-B/14 一图 gallery baseline 结果：validation Top-1 38.73%、Top-3 47.22%；test Top-1 39.14%、Top-3 47.81%。验证/测试差异很小，说明 baseline 稳定，但通用视觉特征对猫个体区分仍明显不足。
- 下一技术优先级应是在 train 身份上进行猫脸领域微调（ArcFace/SupCon 或 proxy-based metric learning），并仅用 validation 选择超参数；不应根据已经查看的 test 结果反复调参。

### 2026-09-14 猫脸高置信错误联系表目视复核

- 已从服务器下载 `validation_hard_errors_contact_sheet.png` 与 `validation_hard_errors.csv`，本地 SHA-256 分别为 `ff4f5531...c3b0`、`62daaace...77fd`，与服务器一致。
- 前 20 条高置信错误中，多条查询图与预测身份 gallery 肉眼近似同一张照片或同一只猫；`018964` 与 `032051` 还出现互相误认，`049181` 与 `049178` 编号相邻且外观高度一致。这批样本优先指向跨身份近重复、身份拆分或原始标签冲突，不能简单归因于模型能力不足。
- 其余主要困难包括幼猫成长阶段差异、强侧脸/仰视、闭眼、模糊、光照和主体尺度变化；少数花色相近的幼猫也存在客观细粒度混淆。
- 当前联系表使用方形居中裁剪展示，适合人工对照，但这不等于训练/评测阶段已完成猫脸检测或关键点对齐。
- 下一实验应先在 validation 上量化跨身份近重复和疑似标签冲突，再在排除或单独标记这些样本的固定协议上比较原图特征与猫脸区域裁剪；test 集保持冻结。
- validation 全量 60,292 张图的 64 位 dHash 审计发现 349 组跨身份近重复候选，涉及 418 个身份；距离 0/1/2/3/4 的候选分别为 98/42/32/40/137 组。98 组距离 0 候选的 SHA-256 均不同，说明不是字节完全相同，而是视觉内容几乎相同的重编码、缩放或轻微处理版本。
- 前 20 个高置信误认身份对中，10 对被 dHash 距离不超过 4 的候选直接命中，包括 `098204↔105717`、`049181↔049178`、`018964↔032051`、`079977↔082255`、`074745↔066795`、`033743↔024235`、`058975↔055662`、`062566↔066795`、`090352↔058311`。这证明至少部分“模型错误”实际受跨身份近重复或身份冲突影响。
- 原冻结 DINOv2 预处理将任意长宽图片直接拉伸为 224×224。改为保持比例的 Resize(256)+CenterCrop(224) 后，validation 一图 gallery Top-1/Top-3 为 37.66%/46.30%，低于原流程约 38.73%/47.22%，不采用。
- OpenCV frontal-cat-face extended 级联在 validation 检出 41,061/60,292 张，覆盖率 68.10%。在检测成功且两侧严格使用相同 11,644 个 gallery 身份、26,931 个 query 的配对子集上，原图 Top-1/Top-3 为 46.99%/56.23%，猫脸裁剪为 44.80%/53.93%，分别下降约 2.19/2.30 个百分点。
- 当前证据不支持中心裁剪或传统 Haar 猫脸裁剪。下一优先级是先治理 validation 的跨身份近重复/标签冲突；未来若重试裁剪，需要使用专门标注训练的猫脸检测器，并继续保持 test 冻结。
- 349 对跨身份 dHash 候选经传递关系归并为 178 个关联组，共涉及 418 个身份。已生成逐对复核 CSV 和身份排除 JSON；原始图片、身份目录及标签均未改动，也没有自动合并身份。
- 为避免把感知哈希候选误当作已确认标签错误，采用最保守的敏感性口径：暂时排除所有相关身份。validation 因此从 60,292 张减少到 58,639 张，排除 1,653 张。
- 适配器常规 1/2/3 张登记照 Top-1 从 59.87%/70.23%/76.49% 变为 60.16%/70.58%/76.88%，绝对提升约 0.29/0.35/0.40 个百分点；冻结 DINOv2 对应提升约 0.14/0.35/0.35 个百分点。
- 固定 query 起点的公平口径结论一致：适配器 1/2/3 张登记照 Top-1 分别提升约 0.37/0.36/0.40 个百分点。提升稳定但很小，说明跨身份近重复/疑似标签冲突确实造成干扰，却不是当前 23%–40% Top-1 错误的主要来源。
- 下一步不应继续扩大哈希自动排除范围。应人工确认候选组；模型研究优先回到猫个体特征学习本身，并保持 test 冻结。
- 在保持 DINOv2 冻结的前提下，为 768 维适配器加入输入残差，并把每批采样改为 96 个身份×每身份 3 张。2000 步 validation 单图登记 Top-1/Top-3 为 60.43%/70.30%，旧模型为 59.87%/70.02%，绝对提升约 0.56/0.28 个百分点。
- 固定 8,283 个身份与 16,794 张相同 query 时，新模型 1/2/3 张登记照 Top-1 为 61.88%/71.47%/76.84%，旧模型为 61.36%/71.03%/76.49%，三档分别提升约 0.52/0.43/0.35 个百分点。
- 独立评测入口最初会按旧结构重建模型，无法恢复残差前向逻辑；现已统一从 checkpoint 的 `config.residual` 重建，并通过回归测试。新模型验证结果独立重载后与训练记录完全一致。
- 这轮提升稳定但幅度有限，说明“更多同身份正样本＋保留原特征旁路”有效，但冻结通用特征的上限正在显现。下一轮应比较更强度量损失或谨慎解冻 DINOv2 后部层，不应立即读取 test。

## Visual/Browser Findings

### 2026-09-04 多照片登记工程检查

### 2026-09-05 路径独立性

- 路径规则集中在 `scripts/project_paths.py`：默认资源锚定项目根目录，用户显式相对路径锚定调用目录，registry 同时兼容绝对路径、旧 `data/...` 项目相对路径和 registry 相对路径。
- 模型依赖延迟到 argparse 解析后，缺少 OpenCV/InsightFace 的轻量环境仍可查看帮助和测试路径逻辑；真正推理仍明确需要 face310 环境。
- 服务器从 `/tmp` 直接运行底层识别脚本并成功读取隔离登记库，确认路径修复不是只通过 mock 的本地测试。

### 2026-09-05 开放集阈值评估准备

- 现有 v3 报告只测试已登记身份的剩余照片，不能测量陌生人误接受率，因此不能据此独立选择拒识阈值。
- 初步候选为 `ljj`、`lls`、`test`、`xyy`、`zh`。真实模型发现后三者分别与 `yjr`、`syy`、`xzh` 达到 0.614081、0.621560、0.637131；逐图复核确认是同一人的别名/误分组，不能算陌生人。最终仅 `ljj`、`lls` 两张作为真正陌生查询。
- 已知组仍按每人前 3 张登记、剩余照片查询。阈值扫描仅对成功检测的人脸计算：已知人正确接受率、陌生人 FAR、陌生人拒识率和平衡准确率；检测失败单独报告，不让阈值掩盖检测问题。
- 清洗后样本扫描在 0.30–0.34 均得到：成功检测的已知查询 51/51 正确接受、陌生查询 2/2 正确拒识，扫描器按保守同分规则返回 0.34。0.50 下为已知 48/51、陌生 2/2；另有 2 张已知查询检测失败。
- 只有两张陌生照片，0.34 的表面最优极可能过拟合，不能声称获得生产级安全阈值；默认继续保留 0.50，待收集至少 30–50 位真正陌生人员并拆分校准/测试集后再调整。

### 2026-09-07 新增陌生人样本盘点

- 桌面 `陌生人测试照片` 中有 30 个编号文件夹，其中 1–25 非空、26–30 为空；有效规模为 25 位、158 张 JPEG 照片，无非图片文件。
- 每位照片数为 2–21 张不等；数量足够先做检测质量和身份泄漏检查，空文件夹应自动忽略。
- 为避免把陌生照片复制进已知人员数据集，评估脚本新增独立 `--unknown-dataset` 输入；该目录下所有非空子文件夹只作为陌生查询。
- 服务器实际评估共 211 条查询：53 条已知、158 条陌生。已知查询检测成功 51 条；陌生查询检测成功 150 条，其中 9 条为多脸，严格单脸有效陌生样本 141 条。
- 默认阈值 0.50 下，成功检测的 150 条陌生查询全部拒识，FAR=0；已知查询正确接受 48/51。25 个陌生人组没有发现与已登记人员身份重复的迹象。
- 阈值扫描在 0.34 上得到当前样本的表面最优结果：已知 51/51 正确接受、陌生 150/150 拒识。清洁单脸子集的最高陌生相似度为 0.329052，已知最低相似度为 0.348025，间隔仅约 0.019；样本既参与校准又参与评分，暂不据此下调默认 0.50。
- 8 张未检测到人脸的照片集中在陌生人1（6 张）、陌生人4（1 张）、陌生人10（1 张）；9 张多脸照片来自陌生人9、12、19、21、22。逐图复核显示多脸主要来自背景人物，严格评估应裁剪或排除。
- 首次严格单脸重跑还发现已知登记候选 `zr1.jpg`、`zxy3.jpg` 各检测到 2 张脸。初版严格逻辑将它们丢弃后没有继续补足 `enroll-count=3`，使相关人员只用 2 张有效照片登记；严格评估应按“成功登记满 N 张”切分登记/查询，而不是机械取前 N 个文件。
- 最终严格重跑按“成功收集满 3 张有效单脸”登记：`zr1.jpg`、`zr2.jpg`、`zxy3.jpg` 等多脸登记候选被跳过后继续补足。最终共有 50 张已知查询，其中 48 张有效单脸、2 张无脸；158 张陌生查询中 141 张有效单脸、8 张无脸、9 张多脸。
- 最终报告在阈值 0.50 下得到已知正确接受 46/48、陌生误接受 0/141；样本内 0.32–0.44 均为满分，程序按同分取更高阈值推荐 0.44。有效已知最低相似度 0.446751，陌生最高 0.319720，间隔约 0.127；因同一批数据既校准又评分，默认阈值仍保留 0.50。

- 识别端已有多行 embedding 的最高余弦相似度逻辑，无需更改识别算法。
- CLI 登记已改为仅接受恰好单脸的照片，不再默默选择合照最大脸；不同照片是否同一人仍由使用者确认，不宣称实现自动身份一致性/模糊度检查。
- 写入原子性只保证登记 JSON 替换；未实现跨进程锁或断电事务恢复。不要与其他登记进程并发使用。
- 本地 10 项离线回归通过；后续通过取消超时 SFTP 的旧传输、重连并补传，已完成三文件哈希确认及远程真实模型验收：两张 person1 照片入库、双人照片拒绝、登记照片回查 matched、astronaut 样例 unknown_person、全部拒绝前后隔离库 JSON 哈希一致。仅为流程验收，不代表独立数据集准确率；CPUExecutionProvider 未改变。详见 progress.md。

- 浏览器搜索结果确认：InsightFace 是开源 2D/3D 人脸分析工具箱，适合人脸检测/识别 baseline。
- 浏览器搜索结果确认：Qdrant 和 Milvus 都是主流开源向量数据库，均支持高维向量相似度检索。
- 浏览器搜索结果确认：PetFace 是动物脸个体识别数据集/benchmark，覆盖猫等多种动物。
- 浏览器搜索结果确认：存在 Cat Individual Images 等猫个体图片数据集，可用于猫脸识别初期实验。

---

*本文件用于保存调研依据、技术发现和后续决策。每完成一次重要搜索、实验或技术选择后应更新。*
- 新独立测试集位于桌面 `新测试`，共 9 位、54 张：cx 4、hb 7、nx 19、syy 5、xzh 4、yjr 3、yzh 4、zr 2、zxy 6。原文件只读检查，HEIC/JPEG/PNG 均能生成预览。
- 初步目视：cx 4 张均为清晰单人近景，包含戴眼镜/不戴眼镜及轻微角度变化；hb 7 张均为单人，场景与角度变化较丰富，其中部分为截图或略柔焦，但适合作为真实困难测试样本。
- nx 19 张全部看起来是单人，涵盖正脸、侧脸、暗光、远景、模糊和截图；`IMG_1836` 与 `IMG_1837` 画面近似重复，实际评测宜只保留一张，另有若干照片难度较高但值得保留用于压力测试。
- syy 5 张均为单人，但多数为中远景、脸占画面比例较小；`906fc...` 中人物闭眼且视线偏离。可先测试检测能否通过，若失败应归为采集质量问题，而不是识别错误。
- xzh 4 张均为清晰单人且脸部大小合适，但 `IMG_6003/6004/6005` 是同一场景、姿态非常接近，独立性与多样性较弱；适合测试但最好日后补不同光线/不戴眼镜的新照片。
- yjr 3 张均为清晰单人近景，包含轻微转头与不同场景；数量达到最低测试要求，但其中两张同为耳机场景且稍柔焦。
- yzh 4 张都是单人且场景差异明显；其中两张为全身/远景、脸较小，一张图像方向特殊，是检测能力的有效困难样本，但不宜全部当作标准识别样本。
- zr 仅 2 张，低于建议的最低 3 张；两张均单人，一张明显模糊、另一张为截图式照片。可以先跑，但不足以单独判断该人员识别稳定性。
- zxy 6 张均为单人，覆盖正脸、轻度侧脸、近景和室外场景；一张闭眼、部分照片方向特殊，适合作为独立困难测试。
- 新测试集与本地旧 `data/face_test_by_person` 未发现文件级 SHA-256 完全重复；HEIC 已在临时副本中转换为 JPEG，原文件没有改动。最终仍需模型检测确认单脸有效性。
### 新测试集 v2 独立评测（2026-09-07）

- 数据集包含 9 位已登记人员、54 张新照片；旧登记照片全部排在查询照片之前，避免严格登记扫描把查询照片误当作登记照。
- 54 张查询中 47 张为有效单脸；阈值 0.50 下正确 42/47（89.36%）。4 张无脸、3 张多脸是检测/构图问题，不应通过降低识别阈值解决。
- 5 张有效单脸被拒识：nx 两张相似度约 0.493，zr 一张约 0.405，zxy 两张约 0.464 与 0.390。当前没有证据表明需要重新上传整批照片；只需按需补拍这些困难样本。
- 初版低阈值统计误用了按 0.50 截断后的 `pred_person`；已改为使用 `best_person` 重新计算。结合既有陌生人开放集结果（0.50 下误接受 0/141），默认阈值先保持 0.50，待修正后的阈值比较写入报告。
### 服务器模型权重与人员特征库位置（2026-09-08）

- 实际模型目录为 `/home/cx/.insightface/models/buffalo_l/`，不是项目内的 `models/` 目录；InsightFace 首次加载 `buffalo_l` 时将预训练 ONNX 文件缓存到用户目录。
- 实际文件：`det_10g.onnx`（16,923,827 bytes，人脸检测）、`1k3d68.onnx`（14,360,619 bytes，3D 关键点）、`2d106det.onnx`（5,030,888 bytes，2D 关键点）、`w600k_r50.onnx`（174,383,860 bytes，人脸识别/512 维特征，项目最核心的识别权重）、`genderage.onnx`（1,322,532 bytes，性别年龄）。
- 当前默认人员登记索引为 `/home/cx/face_recognition_baseline/data/registry.json`，服务器实测大小 352 bytes；其引用的特征文件为 `/home/cx/face_recognition_baseline/data/people/person_001/embeddings.npy`。
- 9 人分组评测使用的是 `data/face_test_by_person/` 的照片临时提取特征，并不等于这些人都已写入默认 `registry.json`；默认库目前只核对到 `person_001`。

### 正式人员特征库现状（2026-09-08）

- 默认人员库现已正式包含 9 位真实人员及 1 条历史 smoke-test，共 10 条记录；真实人员编号连续为 `person_002` 至 `person_010`。
- 9 人共保存 75 个 512 维 embedding：cx 8、hb 7、nx 13、syy 16、xzh 11、yjr 4、yzh 6、zr 3、zxy 7。登记时拒绝无脸/多人照片，不需要为此重传整批数据。
- 正式库已通过独立照片端到端验证：新 yzh 照片正确命中 `person_008`，相似度 0.981546；说明模型权重、默认 registry 和多向量人员特征读取链路均正常。

### 一条命令入口（2026-09-09）

- `recognize.sh` 只负责参数检查、自动激活 `face310` 和调用 `face_cli.py recognize`，没有复制检测、embedding 或相似度算法，因此后续维护仍只有一套识别逻辑。
- 用户实际使用时只需 `bash recognize.sh 图片路径`；第二个可选参数为阈值，默认 0.50。输出重点读取 `best_match_name`、`similarity` 和 `decision`。

### 猫脸域适配结论（2026-09-10）

- 冻结通用视觉特征并仅训练小型监督式对比学习投影头，已经能把“单张 gallery 识别同一只猫”的 test Top-1 从 39.14% 提升到 59.79%，说明主要瓶颈之一确实是特征空间未针对猫个体身份优化，而不只是检索实现问题。
- validation 59.87% 与 test 59.79% 非常接近，当前没有明显验证集过拟合迹象；但 40.21% 的 Top-1 错误仍然较高，当前结果属于有效 baseline，不应称为可部署的高精度猫脸识别系统。
- 下一步应优先在 validation 上比较多图 gallery、图片质量/每身份样本数分层与猫脸区域裁剪；保留 test 不再参与模型或阈值选择。
- validation 公平对照（固定 8,283 个身份、16,794 张完全相同的 query）证明多图登记本身有显著收益：1→2 张使 Top-1 从 61.36% 升至 71.03%，2→3 张升至 76.49%。因此近期 MVP 应默认每只猫至少登记 2 张、推荐 3 张不同姿态/场景照片；收益递减但仍明显。
- 三张登记照协议的错误并非平均分散：3,949 个错误来自 2,880 个身份，1,291 个身份的所有剩余 query 均失败，适合优先检查这些身份的标签一致性、视角变化和主体占比。
- 拥有 7 张以上照片的 validation 身份 Top-1 为 80.43%，优于 4–6 张组约 74.09%–76.88%；更多且更丰富的采集有帮助，但 5 张组反而略差于 4 张组，说明不能把张数当成唯一质量指标。
