# 人脸与猫脸个体识别工程

当前工作区包含两个相互独立但方法相近的子工程：

- 根目录：基于 InsightFace 的人脸识别 baseline、人员登记与开放集评测。
- [`cat-recognition-system/`](cat-recognition-system/README.md)：基于 DINOv2 embedding 和监督式对比学习适配器的猫脸个体识别实验。

大体积数据、模型权重和 embedding 保存在服务器；本地工程保存代码、说明和小型评测报告。

## 本地目录

```text
人脸识别/
├── face_cli.py                 # 人脸识别统一命令入口
├── recognize.sh                # 服务器一条命令识别
├── scripts/                    # 人脸检测、登记、识别与评测实现
├── tests/                      # 人脸工程回归测试
├── data/                       # 本地人脸测试数据，不纳入 Git
├── reports/                    # 本地评测报告，不纳入 Git
├── docs/                       # 使用说明与学习路线
├── cat-recognition-system/     # 猫脸个体识别子工程
├── task_plan.md                # 当前路线
├── progress.md                 # 实际执行记录
└── findings.md                 # 实验结论
```

## 人脸识别 baseline

这是一个基于 InsightFace `buffalo_l` 模型的最小人脸识别项目，包含人脸检测、特征提取、相似度比较、人员登记、人员识别和分组数据评估。

## 当前能力

- 检测照片中的人脸。
- 输出 512 维 embedding。
- 比较两张人脸的余弦相似度。
- 把人员登记到本地 NumPy/JSON 数据库。
- 识别已登记人员或拒识为陌生人员。
- 批量评估按人员文件夹分组的照片。

## 远程环境

```bash
ssh -p 2006 cx@49.232.174.36
cd ~/face_recognition_baseline
source ~/miniconda3/etc/profile.d/conda.sh
conda activate face310
```

## 统一命令入口

### 最简单的实际识别

在服务器上只需要提供一张图片；脚本会自动进入 `face310` 环境，并使用默认阈值 `0.50`：

```bash
cd ~/face_recognition_baseline
bash recognize.sh /照片的完整路径/photo.jpg
```

输出中的 `best_match_name` 是最相似的人员姓名，`similarity` 是相似度，`decision=matched` 表示确认命中，`decision=unknown_person` 表示按阈值拒识。需要实验其他阈值时，可以把阈值作为第二个参数，例如 `bash recognize.sh photo.jpg 0.55`。

查看帮助：

```bash
python face_cli.py --help
```

统一入口和底层脚本的默认数据、分组数据集及报告路径都以项目根目录为基准，因此可以从其他工作目录启动。命令中显式提供的相对路径仍以执行命令时的目录为基准；旧 `registry.json` 中以 `data/...` 保存的相对特征路径也继续兼容。

检测一张人脸：

```bash
python face_cli.py demo --image data/samples/astronaut.jpg
```

比较两张照片：

```bash
python face_cli.py demo \
  --image data/samples/person1_a.jpg \
  --image2 data/samples/person1_b.jpg \
  --threshold 0.50
```

登记一位人员（单照片用法保持兼容）：

```bash
python face_cli.py register --name person1 --image data/samples/person1_a.jpg
```

同一次登记使用多张照片：

```bash
python face_cli.py register --name person1 --image data/samples/person1_a.jpg data/samples/person1_b.jpg
```

也可以重复使用 `--image`。每张照片必须恰好有一张人脸；无脸、多脸、缺失、无法读取和不支持格式的图片会被逐张报告并跳过，重复路径只处理一次。至少一张合格才会建立人员记录，全部不合格则退出码为 1，登记库不变。每次调用创建新人员，不会按名字合并旧记录。

每张合格照片保存一条 512 维特征，识别时取与该人员各特征的最高相似度。请自行确认照片属于同一人，并选择清晰、不同角度的照片；当前不自动校验跨照片身份或模糊程度。支持 `--data-dir` 指定独立测试库；不要同时运行多个登记进程或与网页同时写入。

识别一张照片：

```bash
python face_cli.py recognize --image data/samples/person1_b.jpg --threshold 0.50
```

使用当前推荐参数评估分组数据：

```bash
python face_cli.py evaluate
```

`evaluate` 的默认参数为：

- `threshold=0.50`
- `enroll-count=3`
- `exclude=test`
- `min-images=2`
- 报告输出到 `reports/face_eval.csv`

开放集阈值评估要把从未登记的人明确标为陌生组。已放在默认分组数据集中的陌生人可以用 `--unknown-group` 指定；它们只作为陌生查询，不会进入登记库：

```bash
python face_cli.py evaluate \
  --unknown-group ljj \
  --unknown-group lls \
  --output reports/open-set-clean.csv
```

独立收集的陌生人分组目录用 `--unknown-dataset` 传入，目录下每个非空子文件夹代表一位陌生人，不参与登记：

```bash
python face_cli.py evaluate \
  --unknown-dataset /home/cx/unknown-eval-20260907 \
  --strict-single-face \
  --output reports/final25.csv
```

命令同时生成逐图片报告和名称带 `_thresholds` 的阈值扫描报告。扫描指标包含已知人正确接受率、陌生人误接受率（FAR）、陌生人拒识率和两类的平衡准确率。`--strict-single-face` 会把无脸和多脸照片分别记为数据质量失败，不让它们进入阈值统计；登记阶段也会跳过不合格候选，继续扫描直到收集满指定数量的有效单脸照片。2026-09-07 的独立数据共有 25 位、158 张照片：8 张无脸、9 张多脸，严格单脸有效样本为 141 张。最终评测还有 50 张已知查询，其中 48 张为有效单脸。样本内扫描在 `0.32–0.44` 均得到满分，程序按更保守的同分规则推荐 `0.44`；有效已知最低相似度 `0.446751`，陌生人最高相似度 `0.319720`，间隔约 `0.127`。阈值 `0.50` 下陌生人 FAR 为 `0/141`，已知人正确接受为 `46/48`。由于这批数据同时参与了选阈值和评分，不能当作独立最终测试，因此默认阈值继续保留为较保守的 `0.50`。最终报告为 `reports/final25.csv` 和 `reports/final25_thresholds.csv`，问题照片见 `reports/open-set-25_sample_review.csv`。

## 数据流

```text
图片
  → InsightFace 人脸检测
  → 选择最大人脸
  → 提取 512 维 embedding
  → 余弦相似度
  → threshold 阈值判断
  → matched / unknown_person
```

## 已知限制

- 命令行登记拒绝多脸照片；检测和识别命令仍默认选择面积最大的人脸。
- 强侧脸、低头、横向躺姿和遮挡可能导致检测或识别失败。
- 服务器可看到 RTX 3060，但当前 InsightFace 仍使用 `CPUExecutionProvider`。
- `0.50` 是当前小样本上的工程起点，不是生产环境通用阈值。
