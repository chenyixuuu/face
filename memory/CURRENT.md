# 当前工程状态

更新日期：2026-09-15

## 当前目标

继续完善人脸识别与猫脸个体识别工程，保留代码、数据集、报告和后续模型产物的完整版本记录。

## 已完成

- 人脸识别命令行 baseline、登记、识别和分组评估流程已经建立。
- 本地人脸测试数据与开放集评估报告已经保存在工程中。
- 猫脸工程代码和测试目录已经纳入当前工程。
- 工程已迁移到 `/Users/chenxi/Desktop/face`，旧目录暂时保留作备份。
- Git 仓库已经初始化为 `main` 分支，并成功推送到 `git@github.com:chenyixuuu/face.git`。

## 当前阶段

- 人脸识别：核心 baseline 已完成，继续学习代码并完善工程化使用方式。
- 猫脸识别：服务器侧数据审计、划分、validation 清理分析和 Triplet 辅助残差适配器训练已完成；当前最佳 validation 单图登记 Top-1 为 60.59%。

## 下一步

1. validation 的 178 个跨身份候选关联组已完成优先级分层，并目视复核第 133 组；其余 177 组仍待核，不自动修改标签。详见 `cat-recognition-system/VALIDATION_IDENTITY_REVIEW.md`。
2. 若继续提升模型，评估局部解冻 DINOv2，并设置资源预算和 validation 提升门槛；test 保持冻结。
3. 后续从服务器同步新增猫脸训练代码、数据或权重时，先核验文件清单和校验值。

## 最近验证

- 全部 Python 文件通过语法编译检查。
- 人脸模块 23 项单元测试全部通过。
- 猫脸适配器测试需要 PyTorch；当前本机隔离运行环境未安装 PyTorch，因此本次未运行该项测试。
- 猫脸 validation 中心裁剪与传统级联猫脸裁剪均已实测，指标低于原图方案，当前不采用。
- 排除 418 个近重复候选相关身份后，适配器 1/3 张登记照 Top-1 仅提高约 0.29/0.40 个百分点；冲突有影响但不是主要瓶颈。
- 新残差适配器 validation 单图登记 Top-1/Top-3 为 60.43%/70.30%；公平 1/2/3 图登记 Top-1 为 61.88%/71.47%/76.84%。权重位于 `cat-recognition-system/run_artifacts/adapter_residual_i3_v1/best.pt`。
- Triplet 权重 0.20 的新最佳模型 validation Top-1/Top-3 为 60.59%/70.46%；公平 1/2/3 图登记 Top-1 为 62.08%/71.63%/76.99%。权重位于 `cat-recognition-system/run_artifacts/adapter_triplet_w020_v1/best.pt`。
- 队长 ArcFace V2 已用同一原始 validation 公平复评：标准 1/2/3 图 Top-1 为 59.21%/70.02%/77.12%。当前 Triplet 在 1～2 图登记更好；ArcFace V2 在 3 图时只高 0.13 个百分点。报告见 `cat-recognition-system/ARCFACE_V2_COMPARISON.md`，test 未读取。
- 未知猫拒识内部实验：2 图登记、排除 418 个疑似标签冲突 ID，约 1% 未知猫误接受时，已知猫正确接受仅 30.94%。不部署自动身份确认；先做 Top-3 候选与人工审核。详见 `cat-recognition-system/UNKNOWN_CAT_VALIDATION.md`。
- 服务器 `transformer` 环境已增加并固定 `opencv-python-headless==4.10.0.84`，用于本次猫脸级联检测实验。

## 详细历史

长期计划、实验结论与过程记录继续参见根目录的 `task_plan.md`、`progress.md` 和 `findings.md`。
