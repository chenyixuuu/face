# 猫脸个体识别实验

本目录保存猫脸个体识别的训练、评测和分析代码。大体积原始数据、embedding 和模型权重当前仍位于服务器；已下载的小型审计报告和错误联系表保存在本目录的 `run_artifacts/`。

## 当前结果

- 数据清洗后保留 164,100 个猫身份、601,641 张图片。
- 冻结 DINOv2 ViT-B/14 reg4 基线：测试集 Top-1 `39.14%`。
- 监督式对比学习适配器：验证集 Top-1 `59.87%`，测试集 Top-1 `59.79%`。
- 残差适配器（每身份采样 3 张）：验证集单图登记 Top-1/Top-3 提升至 `60.43%/70.30%`；尚未读取测试集。
- 在残差适配器上加入 Batch-hard Triplet Loss（权重 `0.20`）后，验证集单图登记 Top-1/Top-3 进一步提升至 `60.59%/70.46%`；测试集继续冻结。
- 最新模型固定验证身份与查询图片时，使用 1、2、3 张登记照的 Top-1 分别为 `62.08%`、`71.63%`、`76.99%`。
- 未知猫拒识内部实验：2 图登记时，为把未知猫误接受率压至约 1%，已知猫正确接受率仅 `30.94%`；不应自动确认身份。详见 `UNKNOWN_CAT_VALIDATION.md`。
- validation 跨身份感知哈希审计发现 349 组近重复候选，涉及 418 个身份；前 20 个高置信错误身份对中有 10 对被直接命中。
- 349 对候选组成 178 个关联组；保守排除全部 418 个相关身份后，适配器 1/3 张登记照 Top-1 从 `59.87%/76.49%` 升至 `60.16%/76.88%`。提升较小，说明近重复冲突不是当前主要瓶颈。
- 普通中心裁剪使冻结 DINOv2 validation Top-1 从约 `38.73%` 降至 `37.66%`。
- OpenCV 猫脸级联检测覆盖率为 `68.10%`；在相同检测成功子集上，猫脸裁剪 Top-1 `44.80%`，低于原图 `46.99%`，当前不采用。

测试集只用于已经完成的最终评测；后续实验和参数选择继续使用验证集。

## 目录结构

```text
cat-recognition-system/
├── README.md
├── scripts/
│   ├── train_metric_adapter.py       # 训练轻量特征适配器
│   ├── evaluate_metric_adapter.py    # 独立验证/最终测试入口
│   ├── analyze_gallery_sizes.py      # 比较多图登记效果
│   ├── analyze_validation_errors.py  # 统计验证集错误
│   ├── build_validation_exclusions.py # 生成复核与敏感性排除清单
│   └── make_error_contact_sheet.py   # 生成错误图片联系表
└── tests/
    ├── test_metric_adapter.py
    └── test_validation_cleanup.py
```

## 服务器位置

```text
项目：/home/firecom/yjr/cat-recognition-system
清洗数据：/home/firecom/yjr/cat-recognition-system/data_clean_stream_v2
特征与报告：/home/firecom/yjr/cat-recognition-system/run_artifacts
最佳权重：/home/firecom/yjr/cat-recognition-system/run_artifacts/adapter_triplet_w020_v1/best.pt
```

服务器使用 Conda 环境 `transformer`。核心依赖为 PyTorch、NumPy、FAISS 和 Pillow。

## 主要产物

```text
run_artifacts/adapter_supcon_v1/
├── best.pt
├── history.jsonl
├── test_metrics.txt
├── validation_gallery_sizes.json
├── validation_error_summary.json
├── validation_hard_errors.csv
├── validation_merge_candidates.csv
├── validation_excluded_identities.json
├── validation_clean_sensitivity.json
└── validation_hard_errors_contact_sheet.png

run_artifacts/adapter_residual_i3_v1/
├── best.pt
├── last.pt
├── history.jsonl
├── validation_metrics.json
└── validation_gallery_sizes.json

run_artifacts/adapter_triplet_w020_v1/
├── best.pt
├── last.pt
├── history.jsonl
├── screening_summary.json
├── validation_metrics.json
└── validation_gallery_sizes.json
```

## 下一步

1. 人工复核 178 个跨身份候选组，确定哪些是同猫多 ID、近似构图或误报；脚本不会自动改标签。
2. Triplet 辅助损失已带来小幅提升；下一轮若继续追求模型上限，应评估仅解冻主干最后若干层，仍只用 validation 选方案。
3. 若继续研究裁剪，应换用经过猫脸标注训练的检测器；当前中心裁剪和传统级联方案均不采用。
4. 为实际登记流程采用至少 2 张、推荐 3 张不同姿态或场景照片。
